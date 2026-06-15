# src/model.py
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

class LIVRModelManager:
    """
    Trình quản lý cấu hình kiến trúc mạng LIVR:
    1. Khởi tạo Qwen2.5-VL và mở rộng Vocab cho K=16 tokens ẩn.
    2. Cài đặt mạng LoRA Adapters.
    3. Đăng ký Backward Hook đóng băng tuyệt đối toàn bộ bảng nhúng, ngoại trừ Latent Tokens.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", K=16, device="cuda"):
        self.model_id = model_id
        self.K = K
        self.device = device
        self.latent_tokens = [f"<latent_{i}>" for i in range(K)]
        
        print(f"Loading processor & tokenizer for {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Thêm 16 từ khóa đặc biệt đại diện cho trạng thái ẩn
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": self.latent_tokens})
        self.latent_token_ids = self.processor.tokenizer.convert_tokens_to_ids(self.latent_tokens)
        
        # Lưu các mã nhận diện phục vụ cho phần tạo Mask
        self.image_pad_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id
        
        print(f"Loading model weight...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        print(f"Resizing token embeddings to {len(self.processor.tokenizer)}...")
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

    def setup_peft_and_freezing(self, r=16, alpha=32, dropout=0.05):
        """
        Huấn luyện tham số hóa LoRA kết hợp unfreeze khu biệt Latent Tokens.
        """
        print("Configuring PEFT LoRA...")
        # Target các khối tính toán Attention & MLP trong transformer layers
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        print("Freezing base parameters & setup embedding hooks...")
        # 1. Cho phép bảng nhúng cập nhật tham số
        embed_tokens = self.model.get_input_embeddings()
        embed_tokens.weight.requires_grad = True
        
        # 2. Đăng ký Hook kiểm soát Đạo hàm (Zero out non-latent gradients)
        latent_ids_tensor = torch.tensor(self.latent_token_ids, dtype=torch.long)
        
        def make_embedding_hook(ids):
            def hook(grad):
                # Tạo mặt nạ nhị phân động trên thiết bị của gradient
                mask = torch.zeros(grad.size(0), 1, dtype=grad.dtype, device=grad.device)
                mask[ids.to(grad.device)] = 1.0
                # Triệt tiêu toàn bộ gradient của các token gốc
                return grad * mask
            return hook
            
        embed_tokens.weight.register_hook(make_embedding_hook(latent_ids_tensor))
        
        # 3. Đóng băng các thành phần không liên quan
        for name, param in self.model.named_parameters():
            if "lora_" in name or name == "base_model.model.model.embed_tokens.weight":
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"---> Tham số có thể huấn luyện: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
        
        return self.model
