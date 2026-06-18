# train_kaggle.py
"""
LIVR-Mini-Benchmark Standalone Training Script for Kaggle.
This script automates the entire end-to-end training and evaluation process.
To run: python train_kaggle.py
"""
import os
import sys
import json
import gc
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Setup Kaggle secrets & path environment
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print("➔ Loaded HF_TOKEN successfully from Kaggle Secrets!")
except Exception as e:
    print(f"➔ Secrets loading skipped/failed: {e}")
    print("➔ Will rely on pre-existing HF_TOKEN environment variable.")

# Ensure Python knows where to find src/ directory
sys.path.append(os.getcwd())

from src.model import LIVRModelManager
from src.mask_kaggle import patch_model_for_livr
from src.utils import load_and_inspect_livr_dataset, filter_and_deduplicate_pipeline, prepare_vqa_inputs

def main():
    # 2. Load and override config for Kaggle
    with open("config/implement_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Set Kaggle-specific output directories
    config["output_dir"] = "/kaggle/working/checkpoints"
    config["stage1_lr"] = 5e-5
    config["stage2_lr"] = 2e-5

    print("\n================== CONFIGURATION ==================")
    print(json.dumps(config, indent=2))
    print("===================================================\n")

    # 3. Load & preprocess dataset
    CLEANED_DATA_PATH = "/kaggle/working/cleaned_dataset.pt"
    if os.path.exists(CLEANED_DATA_PATH):
        print(f"➔ Loading cleaned dataset from cache: {CLEANED_DATA_PATH}")
        cleaned_dataset = torch.load(CLEANED_DATA_PATH)
        print(f"[SUCCESS] Loaded {len(cleaned_dataset)} clean samples.")
    else:
        print("➔ Cleaned dataset not found. Downloading and processing from scratch...")
        raw_dataset = load_and_inspect_livr_dataset()
        cleaned_dataset = filter_and_deduplicate_pipeline(
            dataset=raw_dataset,
            target_tasks=['livr_counting', 'livr_object_localization', 'livr_jigsaw', 'livr_visual_similarity'],
            samples_per_task=300
        )
        os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
        torch.save(cleaned_dataset, CLEANED_DATA_PATH)
        print(f"[SUCCESS] Prepared and saved {len(cleaned_dataset)} samples.")

    # 4. Initialize model, quantization, and LoRA
    print("➔ Initializing LIVR Model Manager...")
    manager = LIVRModelManager(
        model_id=config["model_id"],
        K=config["K"],
        device="cuda",
        load_in_4bit=config.get("load_in_4bit", True)
    )
    
    model = manager.setup_peft_and_freezing(
        r=config["lora_r"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"]
    )
    processor = manager.processor

    # Patch model for Custom Attention Masking
    patch_model_for_livr(
        model=model,
        latent_token_ids=manager.latent_token_ids,
        image_pad_token_id=manager.image_pad_token_id,
        pad_token_id=manager.pad_token_id
    )

    # Setup AdamW optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config["stage1_lr"],
        weight_decay=config["weight_decay"],
        eps=1e-6
    )

    # 5. Two-Stage Training loop
    scaler = GradScaler()
    gc.collect()
    torch.cuda.empty_cache()

    STAGE1_EPOCHS = config["stage1_epochs"]
    STAGE2_EPOCHS = config["stage2_epochs"]
    TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS
    GRADIENT_ACCUMULATION_STEPS = config["grad_accumulation_steps"]
    STAGE2_LR = config["stage2_lr"]
    checkpoint_dir = config["output_dir"]

    # Recovery check
    stage1_checkpoint_path = os.path.join(checkpoint_dir, "livr_stage1_checkpoint.pt")
    start_epoch = 1

    if os.path.exists(stage1_checkpoint_path):
        print(f"➔ Found Stage 1 checkpoint at: {stage1_checkpoint_path}")
        checkpoint_size_kb = os.path.getsize(stage1_checkpoint_path) / 1024
        if checkpoint_size_kb > 100:
            print("➔ Loading Stage 1 weights and resuming directly from Stage 2...")
            checkpoint = torch.load(stage1_checkpoint_path, map_location="cuda")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            with torch.no_grad():
                model.get_input_embeddings().weight[manager.latent_token_ids].copy_(
                    checkpoint['latent_embeddings'].to("cuda")
                )
            start_epoch = STAGE1_EPOCHS + 1
            print(f"➔ Setting Stage 2 Learning Rate to: {STAGE2_LR}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = STAGE2_LR
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("⚠ Checkpoint size is too small (corrupted 66KB). Starting from Epoch 1...")
    else:
        print("➔ No Stage 1 checkpoint found. Starting training from Epoch 1...")

    model.train()
    epoch_losses = []

    print("\n====== STARTING THE TRAINING LOOP ======")
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        current_stage = 1 if epoch <= STAGE1_EPOCHS else 2
        model.livr_stage = current_stage

        # Learning Rate reduction transition
        if epoch == STAGE1_EPOCHS + 1 and start_epoch <= STAGE1_EPOCHS:
            print(f"\n➔ STAGE TRANSITION: Lowering LR to {STAGE2_LR} for Stage 2...")
            for param_group in optimizer.param_groups:
                param_group['lr'] = STAGE2_LR

        epoch_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(cleaned_dataset, desc=f"Epoch {epoch}/{TOTAL_EPOCHS} (Stage {current_stage})")
        for step, batch in enumerate(progress_bar):
            try:
                inputs = prepare_vqa_inputs(
                    processor=processor,
                    conversation=batch['conversation'],
                    latent_tokens=manager.latent_tokens,
                    device="cuda"
                )

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

                if torch.isnan(loss):
                    print(f"\n[WARNING] NaN loss at epoch {epoch}, step {step}. Skipping batch...")
                    optimizer.zero_grad()
                    del inputs, outputs, loss
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(cleaned_dataset):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                progress_bar.set_postfix({"Loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

                del inputs, outputs, loss
                if step % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\n[WARNING] GPU OOM. Clearing cache and skipping batch...")
                    optimizer.zero_grad()
                    del e
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss = epoch_loss / len(cleaned_dataset)
        epoch_losses.append(avg_loss)
        print(f"➔ Ended Epoch {epoch} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint after Stage 1
        if epoch == STAGE1_EPOCHS:
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            trainable_sd = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names}
            
            torch.save({
                'model_state_dict': trainable_sd,
                'latent_embeddings': model.get_input_embeddings().weight[manager.latent_token_ids].detach().cpu(),
                'latent_token_ids': manager.latent_token_ids
            }, stage1_checkpoint_path)
            print(f"---> Saved Stage 1 checkpoint at: {stage1_checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "livr_mini_checkpoint.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_sd = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names}
    
    torch.save({
        'model_state_dict': trainable_sd,
        'latent_embeddings': model.get_input_embeddings().weight[manager.latent_token_ids].detach().cpu(),
        'latent_token_ids': manager.latent_token_ids
    }, final_checkpoint_path)
    print(f"\n[SUCCESS] Saved final checkpoint at: {final_checkpoint_path}")

    # 6. Save Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='b', label='Training Loss')
    plt.axvline(x=STAGE1_EPOCHS, color='r', linestyle='--', label='Transition to Stage 2')
    plt.title('LIVR-Mini Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plot_path = "/kaggle/working/loss_curve.png"
    plt.savefig(plot_path)
    print(f"➔ Saved training loss curve plot to: {plot_path}")

    # 7. Evaluate Baseline
    print("\n➔ Evaluating performance (20 baseline samples)...")
    eval_samples = cleaned_dataset[:20]
    
    def evaluate(use_lora=True):
        model.eval()
        if use_lora:
            model.livr_stage = 2
        correct, total = 0, 0
        log_entries = []
        with torch.no_grad():
            for i, item in enumerate(eval_samples):
                conv = item['conversation']
                conv_for_generation = [msg for msg in conv if msg["role"] == "user"]
                inputs = prepare_vqa_inputs(
                    processor=processor,
                    conversation=conv_for_generation,
                    latent_tokens=manager.latent_tokens,
                    device="cuda"
                )
                inputs.pop("labels", None)
                if not use_lora:
                    with model.disable_adapter():
                        outputs = model.generate(**inputs, max_new_tokens=10)
                else:
                    outputs = model.generate(**inputs, max_new_tokens=10)
                
                input_len = inputs["input_ids"].shape[1]
                pred_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                target_text = str(conv[1]["content"][0]["text"]).strip()
                
                is_correct = pred_text.lower() == target_text.lower()
                if is_correct:
                    correct += 1
                total += 1
                
                log_entries.append({
                    "index": i + 1,
                    "target": target_text,
                    "predicted": pred_text,
                    "correct": is_correct
                })
                
                if i < 5:
                    suffix = "LIVR" if use_lora else "Base"
                    print(f"   [{suffix} Sample {i+1}] Correct: {target_text} | Pred: {pred_text} | {'MATCH' if is_correct else 'MISS'}")
        
        # Save detailed logs
        suffix = "livr" if use_lora else "base"
        detail_path = f"/kaggle/working/train_eval_details_{suffix}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, ensure_ascii=False, indent=2)
        print(f"➔ Saved sample-by-sample eval logs to: {detail_path}")
        return (correct / total) * 100

    livr_acc = evaluate(use_lora=True)
    base_acc = evaluate(use_lora=False)

    results_text = (
        f"===================================================\n"
        f" EVALUATION REPORT (20 SAMPLES)\n"
        f"===================================================\n"
        f"LIVR Model Accuracy: {livr_acc:.2f}%\n"
        f"Base Model Accuracy: {base_acc:.2f}%\n"
        f"Accuracy Gain:       {livr_acc - base_acc:+.2f}%\n"
        f"===================================================\n"
    )
    print(results_text)
    
    report_path = "/kaggle/working/eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(results_text)
    print(f"➔ Evaluation report saved to: {report_path}")

if __name__ == "__main__":
    main()
