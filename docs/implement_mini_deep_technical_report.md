# Báo cáo kỹ thuật chi tiết phần Implement Mini LIVR

Tài liệu này dùng để giải thích sâu phần implement mini của dự án LIVR-Mini-Benchmark, đặc biệt là cách fine-tune Qwen2.5-VL bằng QLoRA, cách thêm latent tokens, cách áp dụng bottleneck attention mask, cách xử lý dữ liệu `Kkuntal990/LIVR_mixed`, và các lỗi kỹ thuật đã gặp khi chạy trên Kaggle/Colab.

Mục tiêu nghiệm thu của phần Mini là:

> Training the model from scratch on tested datasets for at least 1 epochs.

Trong dự án này, "from scratch" cần được hiểu chính xác là: không dùng checkpoint LIVR đã train trước, bắt đầu từ base pretrained model `Qwen/Qwen2.5-VL-3B-Instruct`, rồi huấn luyện mới các tham số được phép học gồm LoRA adapters và latent token embeddings. Ta không train toàn bộ Qwen từ random initialization, vì điều đó vượt xa tài nguyên Kaggle/Colab và không phù hợp mục tiêu mini.

## 1. Tóm tắt kiến trúc đang implement

Pipeline implement gồm các bước chính:

1. Tải dataset `Kkuntal990/LIVR_mixed`, config `livr_train`.
2. Lọc các tác vụ thị giác mục tiêu như counting, object localization, jigsaw, visual similarity.
3. Khử trùng lặp ảnh bằng perceptual hash.
4. Định dạng dữ liệu thành chat template của Qwen2.5-VL.
5. Load base model Qwen2.5-VL-3B-Instruct ở 4-bit.
6. Thêm `K=16` latent tokens vào tokenizer.
7. Gắn LoRA adapter vào attention và MLP projection layers.
8. Freeze trọng số gốc của Qwen, chỉ train LoRA weights và embedding của latent tokens.
9. Stage 1: train với bottleneck mask, ép thông tin ảnh đi qua latent tokens.
10. Stage 2: mở mask về causal attention thông thường để mô hình phối hợp ảnh gốc và latent tokens.
11. Lưu checkpoint gồm LoRA weights và latent embeddings.

Các file chính:

- [src/model.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py): load Qwen2.5-VL, cấu hình QLoRA, thêm latent tokens, freeze/unfreeze tham số.
- [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py): tạo bottleneck attention mask và monkey-patch forward.
- [src/utils.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/utils.py): tải dataset, lọc dữ liệu, chuẩn bị input cho Qwen2.5-VL.
- [config/implement_config.json](/home/ductien/Documents/LIVR-Mini-Benchmark/config/implement_config.json): siêu tham số huấn luyện.
- [livr-mini-benchmark.ipynb](/home/ductien/Documents/LIVR-Mini-Benchmark/livr-mini-benchmark.ipynb): notebook Kaggle triển khai huấn luyện.
- [train_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/train_kaggle.py): script chạy standalone cùng logic với notebook.

## 2. QLoRA là gì, khác LoRA thường ở đâu?

### 2.1 LoRA thường

LoRA, viết tắt của Low-Rank Adaptation, là kỹ thuật fine-tune tham số hiệu quả. Thay vì cập nhật toàn bộ ma trận trọng số lớn `W` của mô hình, LoRA đóng băng `W` và thêm một nhánh cập nhật nhỏ:

```text
W' = W + Delta W
Delta W = B @ A
```

Trong đó:

- `W` là trọng số gốc của model, bị freeze.
- `A` và `B` là hai ma trận hạng thấp được train.
- `r` là rank của LoRA, tức chiều nén trung gian của `A` và `B`.
- Vì `r` nhỏ hơn rất nhiều so với kích thước thật của `W`, số tham số cần train giảm mạnh.

Nếu train full model, ta phải cập nhật hàng tỷ tham số. Với LoRA, ta chỉ cập nhật vài chục triệu tham số adapter.

### 2.2 QLoRA

QLoRA là LoRA trên base model đã được quantize 4-bit.

Trong code hiện tại, base Qwen2.5-VL được load bằng:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

Ý nghĩa:

- `load_in_4bit=True`: trọng số gốc của Qwen được lưu trong VRAM ở dạng 4-bit.
- `bnb_4bit_quant_type="nf4"`: dùng NormalFloat4, định dạng 4-bit phù hợp với phân phối trọng số neural network.
- `bnb_4bit_use_double_quant=True`: lượng tử hóa thêm các scale/quantization constants, giảm thêm bộ nhớ.
- `bnb_4bit_compute_dtype=torch.float16`: khi tính toán, tensor được dequantize tạm sang float16 để dùng Tensor Cores của GPU T4.

Vì vậy, tên đúng của kỹ thuật ta dùng là **QLoRA**:

```text
Base model 4-bit + frozen
LoRA adapters trainable
Latent token embeddings trainable
```

Nếu thầy hỏi "em dùng LoRA hay QLoRA?", câu trả lời nên là:

> Em dùng QLoRA. Cụ thể, backbone Qwen2.5-VL-3B được load ở 4-bit bằng bitsandbytes, sau đó em gắn LoRA adapter vào các projection layers. Trong quá trình huấn luyện, trọng số gốc 4-bit không cập nhật, chỉ LoRA weights và latent token embeddings được cập nhật.

## 3. Ý nghĩa các tham số trong config

File `config/implement_config.json`:

```json
{
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "K": 16,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "stage1_epochs": 2,
  "stage2_epochs": 3,
  "stage1_lr": 5e-5,
  "stage2_lr": 2e-5,
  "weight_decay": 0.01,
  "effective_batch_size": 8,
  "batch_size_per_device": 1,
  "grad_accumulation_steps": 8,
  "dataset_name": "Kkuntal990/LIVR_mixed",
  "load_in_4bit": true
}
```

### `model_id`

`Qwen/Qwen2.5-VL-3B-Instruct` là backbone vision-language model. Nó đã có khả năng hiểu ảnh và text trước khi ta fine-tune.

Lý do chọn bản 3B:

- Đủ mạnh cho VQA và visual reasoning.
- Nhỏ hơn các model 7B/13B nên có thể chạy trên Kaggle/Colab.
- Phù hợp với mục tiêu mini benchmark.

### `K = 16`

Đây là số latent tokens:

```text
<latent_0>, <latent_1>, ..., <latent_15>
```

Mỗi latent token là một token đặc biệt được thêm vào tokenizer. Mục tiêu của nó không phải để sinh ra câu trả lời trực tiếp, mà để đóng vai trò bộ nhớ ẩn, nén thông tin ảnh trong Stage 1.

Vì sao không dùng quá ít hoặc quá nhiều?

- Nếu `K` quá nhỏ, bottleneck quá hẹp, không đủ chứa thông tin ảnh.
- Nếu `K` quá lớn, bottleneck yếu đi, attention bị phân tán, mô hình có thể không học biểu diễn cô đọng.
- `K=16` là lựa chọn cân bằng, bám theo tinh thần LIVR.

### `lora_r = 16`

`r` là rank của LoRA. Với mỗi weight matrix lớn, LoRA học hai ma trận nhỏ có rank `r`.

Rank càng cao:

- Năng lực học càng mạnh.
- Số tham số trainable càng nhiều.
- Tốn VRAM và dễ overfit hơn.

Rank càng thấp:

- Nhẹ hơn.
- Nhưng có thể thiếu năng lực thích nghi.

`r=16` là mức hợp lý cho mini benchmark: đủ linh hoạt nhưng vẫn chạy được trên GPU hạn chế.

### `lora_alpha = 32`

`alpha` là hệ số scale của LoRA. Thường LoRA update được scale theo:

```text
scale = alpha / r
```

Với `alpha=32`, `r=16`, ta có:

```text
scale = 32 / 16 = 2
```

Nghĩa là tín hiệu cập nhật từ LoRA được nhân hệ số 2 trước khi cộng vào đường forward. Nếu `alpha` quá nhỏ, adapter học yếu. Nếu quá lớn, training dễ dao động.

### `lora_dropout = 0.05`

Dropout áp vào nhánh LoRA trong training. Mục đích:

- Giảm overfit trên tập mini.
- Làm adapter không phụ thuộc quá mạnh vào một vài feature.
- Tăng ổn định tổng quát.

`0.05` là mức nhẹ, không làm hỏng learning signal.

### `stage1_epochs = 2`, `stage2_epochs = 3`

Huấn luyện gồm 5 epoch:

- Stage 1: 2 epoch với bottleneck mask.
- Stage 2: 3 epoch với causal attention thường.

Tỷ lệ 2:3 tương ứng 40% Stage 1 và 60% Stage 2. Ý nghĩa:

- Stage 1 ép latent tokens học cách chứa thông tin ảnh.
- Stage 2 cho mô hình phối hợp cả ảnh gốc và latent tokens, tránh bị bó buộc quá mức.

Với yêu cầu "at least 1 epoch", notebook đã vượt yêu cầu vì chạy tổng 5 epoch trong log cũ.

### `stage1_lr = 5e-5`

Learning rate Stage 1 cao hơn vì lúc đầu latent token embeddings và LoRA adapter còn chưa học gì. Cần tín hiệu đủ mạnh để bắt đầu hình thành bottleneck representation.

### `stage2_lr = 2e-5`

Learning rate Stage 2 thấp hơn vì đây là giai đoạn tinh chỉnh. Nếu giữ LR quá cao, mô hình có thể làm hỏng latent representation vừa học ở Stage 1.

### `weight_decay = 0.01`

Weight decay là regularization cho AdamW, giúp hạn chế trọng số tăng quá lớn. Với tập mini, điều này giúp giảm overfit.

### `batch_size_per_device = 1`

Vì Qwen2.5-VL xử lý cả ảnh, text, visual tokens và transformer activations, batch lớn sẽ dễ OOM. Batch vật lý bằng 1 là lựa chọn an toàn trên T4/P100.

### `grad_accumulation_steps = 8`

Thay vì update sau mỗi batch, ta cộng dồn gradient qua 8 step:

```text
effective_batch_size = batch_size_per_device * grad_accumulation_steps
                     = 1 * 8
                     = 8
```

Lợi ích:

- VRAM chỉ chịu batch 1.
- Optimizer vẫn thấy tín hiệu giống batch 8.
- Training ổn định hơn batch 1 thuần.

### `load_in_4bit = true`

Đây là điểm biến LoRA thành QLoRA. Nếu tắt option này, model sẽ load 16-bit hoặc bfloat16, dễ vượt VRAM khi train.

## 4. Target modules của LoRA có ý nghĩa gì?

Trong `src/model.py`, LoRA được gắn vào:

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

Các module này thuộc hai nhóm chính.

### Nhóm attention projection

- `q_proj`: tạo Query.
- `k_proj`: tạo Key.
- `v_proj`: tạo Value.
- `o_proj`: projection đầu ra của attention.

Attention là nơi mô hình quyết định token nào nhìn token nào. Vì LIVR can thiệp mạnh vào attention flow, việc train LoRA ở các projection này rất quan trọng.

Nếu thầy hỏi "tại sao không chỉ train MLP?", trả lời:

> Vì bài toán của LIVR nằm ở cách điều hướng thông tin thị giác qua latent tokens. Attention projection là nơi trực tiếp quyết định luồng thông tin giữa image tokens, prompt tokens, latent tokens và answer tokens.

### Nhóm MLP projection

- `gate_proj`
- `up_proj`
- `down_proj`

MLP layers xử lý biến đổi phi tuyến sau attention. Train LoRA ở MLP giúp mô hình học cách chuyển đổi biểu diễn latent thành tín hiệu trả lời tốt hơn.

Nếu chỉ train attention mà không train MLP, mô hình có thể nhìn đúng chỗ nhưng chưa chắc chuyển được thông tin đó thành answer distribution tốt.

## 5. Vì sao chỉ train LoRA weights và latent token embeddings?

Trong `src/model.py`, sau khi gắn LoRA:

```python
for name, param in self.model.named_parameters():
    if "lora_" in name or param is embed_tokens_weight:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

Nghĩa là:

- Trọng số gốc Qwen bị freeze.
- LoRA weights được train.
- Bảng embedding được mở train, nhưng có hook để chỉ latent rows nhận gradient.

Lý do:

1. Full fine-tuning Qwen2.5-VL-3B cần rất nhiều VRAM.
2. Tập mini nhỏ, full fine-tuning dễ overfit.
3. Ta muốn giữ kiến thức ngôn ngữ-thị giác gốc của Qwen.
4. Mục tiêu nghiên cứu là kiểm tra cơ chế latent bottleneck, không phải train lại toàn bộ backbone.

## 6. Embedding hook chống "token drift"

Khi thêm latent tokens, ta cần học embedding của các token mới. Nhưng embedding layer là một ma trận lớn chứa toàn bộ vocabulary. Nếu mở `requires_grad=True` cho cả embedding matrix, gradient có thể cập nhật cả các token gốc như `the`, `cat`, `answer`, `A`, `B`, v.v.

Điều này gây nguy cơ token drift: mô hình bị lệch nghĩa từ vựng gốc.

Giải pháp trong `src/model.py`:

```python
def make_embedding_hook(ids):
    def hook(grad):
        mask = torch.zeros(grad.size(0), 1, dtype=grad.dtype, device=grad.device)
        mask[ids.to(grad.device)] = 1.0
        return grad * mask
    return hook
```

Hook này tạo một mask theo hàng embedding:

- Hàng thuộc latent token id: giữ gradient.
- Hàng thuộc vocabulary gốc: nhân gradient với 0.

Nếu thầy hỏi "vì sao không tạo embedding riêng cho latent tokens?", có thể trả lời:

> Vì Qwen tokenizer và embedding pipeline đã mong đợi token id nằm trong cùng vocabulary. Thêm special tokens rồi resize embedding giúp latent tokens đi qua cùng cơ chế embedding như token thường, đồng thời hook đảm bảo chỉ các hàng latent được học.

## 7. Vì sao trainable params phải cast về fp32?

Trong QLoRA, backbone 4-bit, compute float16. Một số trainable tensors như LoRA hoặc embedding mới có thể ở float16. Khi dùng GradScaler, PyTorch không cho unscale FP16 gradients trực tiếp trong một số cấu hình, gây lỗi:

```text
ValueError: Attempting to unscale FP16 gradients.
```

Giải pháp trong `src/model.py`:

```python
for name, param in self.model.named_parameters():
    if param.requires_grad and param.dtype != torch.float32:
        param.data = param.data.float()
```

Ý nghĩa:

- Base model vẫn 4-bit/frozen.
- Chỉ trainable params được đưa về float32.
- GradScaler và AdamW ổn định hơn.
- Tốn thêm rất ít VRAM vì trainable params nhỏ hơn nhiều so với toàn model.

Đây là một điểm kỹ thuật rất quan trọng nếu thầy hỏi vì sao QLoRA vẫn có tensor fp32.

## 8. Luồng dữ liệu từ dataset vào model

### 8.1 Dataset nguồn

Dataset dùng là `Kkuntal990/LIVR_mixed`, config `livr_train`. Theo metadata Hugging Face, config này gồm:

- `train/metadata.jsonl`
- `train/images/**`

Link dataset: <https://huggingface.co/datasets/Kkuntal990/LIVR_mixed/viewer/livr_train>

Trong `src/utils.py`, code tải bằng hai hướng:

1. Ưu tiên `snapshot_download` để kéo thư mục `train/**` về local.
2. Nếu lỗi, fallback sang:

```python
load_dataset("Kkuntal990/LIVR_mixed", "livr_train", cache_dir=cache_dir)
```

### 8.2 Vì sao dùng snapshot_download?

Khi đọc image dataset lớn, `load_dataset` có thể lazy-load ảnh qua nhiều request nhỏ. Trên Kaggle/Colab dễ gặp:

- tải chậm,
- HTTP 429 rate limit,
- timeout giữa chừng,
- lỗi schema nếu loader tự quét nhiều thư mục metadata khác nhau.

`snapshot_download` giúp tải trước toàn bộ thư mục train về SSD local. Sau đó ta đọc `metadata.jsonl` trực tiếp và ánh xạ `file_name` sang đường dẫn ảnh local.

Nếu thầy hỏi "vì sao không dùng load_dataset thẳng?", trả lời:

> Em vẫn có fallback `load_dataset`, nhưng ưu tiên snapshot để giảm bottleneck mạng và tránh lazy-loading từng ảnh trong vòng lặp tiền xử lý. Với dataset nhiều ảnh, tải một lần về SSD local giúp train ổn định hơn trên Kaggle.

### 8.3 Lọc task và khử trùng lặp ảnh

Trong `filter_and_deduplicate_pipeline`, ta lọc các task mục tiêu:

```python
[
    "livr_counting",
    "livr_object_localization",
    "livr_jigsaw",
    "livr_visual_similarity"
]
```

Sau đó dùng perceptual hash:

```python
v_hash = imagehash.phash(image_obj)
```

Mục tiêu:

- Loại ảnh trùng hoặc gần trùng.
- Giảm nguy cơ mô hình học thuộc ảnh.
- Tạo tập mini sạch và cân bằng.

Lưu ý quan trọng để báo cáo:

- Notebook `livr-mini-benchmark.ipynb` hiện truyền `samples_per_task=700`.
- Output cũ trong notebook cho thấy lần chạy đã dùng `1200` mẫu, tương ứng 300 mẫu/task.
- Nếu cần báo cáo chính xác, nên chốt theo log thực nghiệm: lần nghiệm thu đã train 1200 mẫu, 5 epoch. Nếu rerun sạch với code hiện tại, cần kiểm tra lại số mẫu thực tế sau lọc.

### 8.4 Định dạng conversation

Mỗi mẫu được chuyển thành:

```python
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": abs_image_path},
            {"type": "text", "text": query}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": answer}
        ]
    }
]
```

Đây là format phù hợp với Qwen2.5-VL chat template.

## 9. Hàm `prepare_vqa_inputs` làm gì?

`prepare_vqa_inputs` là cầu nối giữa dữ liệu thô và model.

Các bước:

1. Copy conversation để không sửa dữ liệu gốc.
2. Chèn latent tokens vào cuối prompt user.
3. Áp dụng Qwen chat template.
4. Load ảnh bằng PIL khi cần.
5. Gọi processor để tạo tensor.
6. Tạo labels và mask loss cho phần prompt.

### 9.1 Chèn latent tokens

```python
latent_str = "".join(latent_tokens)
content_item["text"] = f"{content_item['text'].strip()}\n{latent_str}"
```

Latent tokens được đặt sau câu hỏi. Lý do:

- Latent tokens nhìn được prompt trước đó trong causal sequence.
- Chúng biết câu hỏi đang yêu cầu thông tin gì.
- Từ đó học nén thông tin ảnh có điều kiện theo task.

Nếu đặt latent trước prompt, latent chưa biết câu hỏi nên nén ảnh kém định hướng hơn.

### 9.2 Labels chỉ tính loss trên answer

```python
labels = inputs["input_ids"].clone()
prompt_len = prompt_inputs["input_ids"].size(1)
labels[:, :prompt_len] = -100
```

Trong causal LM training, loss mặc định tính trên mọi token. Nhưng ở VQA, ta chỉ muốn model học sinh answer, không học lại prompt.

`-100` là ignore index của CrossEntropyLoss trong PyTorch. Các token prompt, image placeholders, latent tokens không đóng góp loss trực tiếp.

Nếu thầy hỏi "latent tokens có loss không?", trả lời:

> Không có supervised loss trực tiếp trên latent tokens. Chúng học gián tiếp qua loss của answer tokens. Vì answer muốn đúng thì thông tin ảnh phải đi qua latent trong Stage 1.

## 10. Bottleneck attention mask trong Stage 1

Đây là lõi của LIVR.

Trong chuỗi input có các vùng:

```text
[image tokens] [prompt tokens] [latent tokens] [answer tokens]
```

Stage 1 áp mask:

- Prompt tokens không được attend trực tiếp vào image tokens.
- Answer tokens không được attend trực tiếp vào image tokens.
- Latent tokens được attend image tokens.
- Answer tokens được attend latent tokens theo chiều causal.

Do đó đường thông tin ảnh hợp lệ là:

```text
Answer attend Latent attend Image
```

Không cho:

```text
Answer attend Image trực tiếp
Prompt attend Image trực tiếp
```

Mục đích:

> Nếu không chặn đường trực tiếp, mô hình sẽ bỏ qua latent tokens vì ảnh gốc đã đủ trả lời. Bottleneck mask buộc latent tokens trở thành nơi chứa thông tin thị giác cần thiết.

## 11. Stage 2 khác Stage 1 ở đâu?

Stage 2 bỏ bottleneck mask và quay lại causal attention tiêu chuẩn.

Vì sao cần Stage 2?

- Stage 1 ép latent học thông tin ảnh nhưng có thể quá khắt khe.
- Stage 2 cho model nhìn lại ảnh gốc và latent cùng lúc.
- Model học phối hợp visual tokens chi tiết và latent tokens đã cô đọng.

Nếu chỉ Stage 1, mô hình có thể mất chi tiết ảnh. Nếu chỉ Stage 2, latent tokens có thể bị bỏ qua. Hai giai đoạn bổ sung cho nhau.

## 12. Monkey-patching forward có vai trò gì?

Qwen2.5-VL không có API sẵn để truyền bottleneck mask theo logic LIVR. Vì vậy, `src/mask_kaggle.py` monkey-patch hàm `forward`:

```python
model.forward = types.MethodType(livr_forward, model)
```

Trong `livr_forward`:

1. Lấy `input_ids`, `attention_mask`, `position_ids`, `labels`.
2. Nếu `model.livr_stage == 1` và đang training, tạo custom 4D attention mask.
3. Tính `position_ids`/RoPE trước khi ghi đè attention mask.
4. Gọi lại `original_forward`.

Điểm nhạy cảm:

- Transformers version thay đổi có thể đổi signature forward.
- Qwen2.5-VL dùng M-RoPE cho image/video/text positions.
- Nếu truyền sai field như `mrope_position_ids` vào version không hỗ trợ, sẽ lỗi.

Do đó patch hiện tại có logic inspect signature để chỉ truyền tham số mà forward hỗ trợ.

## 13. Vì sao phải xử lý position_ids/RoPE?

Qwen2.5-VL dùng multi-modal rotary position embedding. Với input có ảnh, text, image grid, model cần position ids đúng cho từng modality.

Nếu ta thay `attention_mask` 2D mặc định bằng custom mask 4D quá sớm, hàm nội bộ có thể không còn tính được RoPE position như bình thường. Vì vậy patch phải:

1. Dùng attention mask gốc để tính rope index.
2. Lưu `position_ids`.
3. Sau đó mới thay attention mask bằng bottleneck mask.

Nếu không làm đúng, dễ gặp lỗi generate/forward liên quan `mrope_position_ids`, `rope_deltas`, hoặc shape mismatch.

## 14. Các kỹ thuật vượt giới hạn phần cứng

### 14.1 4-bit quantization

Giảm bộ nhớ lưu trọng số backbone. Nếu model 3B ở fp16 cần khoảng 6GB chỉ cho weight, 4-bit giúp giảm lớn phần này.

### 14.2 LoRA/QLoRA

Chỉ train adapter nhỏ thay vì toàn bộ model. Điều này giảm:

- gradient memory,
- optimizer states,
- checkpoint size,
- nguy cơ overfit.

### 14.3 Gradient checkpointing

Trong forward, transformer thường lưu activations để backward. Gradient checkpointing không lưu hết, mà recompute một phần trong backward.

Đổi lại:

- Tiết kiệm VRAM.
- Chậm hơn một chút.

Trên T4, đây là đánh đổi hợp lý.

### 14.4 Mixed precision + GradScaler

Forward chạy trong:

```python
torch.amp.autocast("cuda", dtype=torch.float16)
```

Float16 nhanh và nhẹ hơn fp32, nhưng dễ underflow gradient. GradScaler scale loss lên trước backward rồi unscale trước optimizer step.

### 14.5 Gradient accumulation

Batch thật bằng 1, tích lũy 8 bước. Đây là cách mô phỏng batch lớn mà không tăng VRAM đỉnh.

### 14.6 Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
```

Giới hạn norm gradient để tránh explosion, đặc biệt quan trọng khi có custom mask và fp16.

### 14.6.1 Giá trị âm hữu hạn trong attention mask

Trong `src/mask_kaggle.py`, vị trí bị chặn trong attention mask được gán `-10000.0` thay vì `-inf`.

Lý do:

- Trong softmax attention, số âm rất lớn làm xác suất gần bằng 0, tức là đã chặn được attention.
- Dùng `-inf` hoặc số quá gần giới hạn dưới của fp16 dễ sinh NaN khi cộng với attention scores.
- `-10000.0` đủ mạnh để chặn nhưng ổn định hơn trên GPU T4 với float16.

### 14.7 Giới hạn pixel đầu vào

Trong `src/model.py`:

```python
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
```

Qwen2.5-VL biến ảnh thành visual tokens. Ảnh càng lớn, visual tokens càng nhiều, attention càng tốn bộ nhớ. Giới hạn pixel giúp kiểm soát số visual tokens.

### 14.8 Lazy image loading

Dataset sau xử lý lưu đường dẫn ảnh thay vì PIL image object. Ảnh chỉ được mở khi batch cần.

Lợi ích:

- Giảm RAM CPU.
- File `.pt` nhẹ hơn.
- Tránh serialize đối tượng ảnh lớn.

### 14.9 Chủ động giải phóng VRAM

Trong vòng lặp train:

```python
del inputs, outputs, loss
gc.collect()
torch.cuda.empty_cache()
```

Điều này giảm nguy cơ VRAM fragmentation và OOM trong notebook dài.

## 15. Checkpoint được lưu như thế nào?

Checkpoint không lưu toàn bộ Qwen 3B. Code chỉ lưu trainable parameters:

```python
trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
trainable_sd = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names}
```

Và lưu thêm:

```python
{
    "model_state_dict": trainable_sd,
    "latent_embeddings": ...,
    "latent_token_ids": ...
}
```

Lý do:

- Base Qwen có thể load lại từ Hugging Face.
- LoRA weights và latent embeddings là phần ta train.
- Checkpoint nhẹ hơn rất nhiều.

Lỗi từng gặp:

- Checkpoint chỉ khoảng 66KB, nghĩa là lọc sai key hoặc không lưu đúng trainable weights.
- Khắc phục bằng cách lấy `trainable_names` từ `model.named_parameters()` rồi lọc `state_dict` theo đúng key.

Nếu thầy hỏi "tại sao không lưu full model?", trả lời:

> Vì backbone là pretrained model cố định, không thay đổi. Lưu full model vừa nặng vừa không cần thiết. Em chỉ cần lưu phần delta đã học: LoRA weights và latent embeddings.

## 16. Các lỗi đã gặp và cách khắc phục

### 16.1 CUDA OOM

Triệu chứng:

```text
RuntimeError: CUDA out of memory
```

Nguyên nhân:

- Model VLM 3B lớn.
- Ảnh sinh nhiều visual tokens.
- Attention mask và activations tốn VRAM.

Khắc phục:

- QLoRA 4-bit.
- Batch size 1.
- Gradient accumulation.
- Gradient checkpointing.
- Giới hạn `max_pixels`.
- `del` tensor và `empty_cache`.
- Skip batch nếu OOM.

### 16.2 HTTP 429 khi tải dataset

Triệu chứng:

```text
HTTP Error 429 thrown while requesting HEAD ...
Rate limited.
```

Nguyên nhân:

- Hugging Face bị rate limit do nhiều request ảnh.
- Dataset ảnh lớn.

Khắc phục:

- Dùng `HF_TOKEN`.
- Dùng `snapshot_download` với `allow_patterns=["train/**"]`.
- Cache `cleaned_dataset.pt` để không tải lại mỗi lần.

### 16.3 Schema mismatch khi load dataset

Nguyên nhân:

- Dataset repo có nhiều split/config với metadata khác nhau.
- Nếu loader tự quét nhiều thư mục, các cột có thể không đồng nhất.

Khắc phục:

- Chỉ đọc đúng `train/metadata.jsonl`.
- Fallback đúng config `livr_train`.

### 16.4 Device mismatch

Triệu chứng:

```text
Expected all tensors to be on the same device
```

Nguyên nhân:

- Custom mask tạo tensor trên CPU trong khi input ở CUDA.

Khắc phục:

- Mọi `torch.ones`, `torch.zeros`, `torch.arange` trong mask đều truyền `device=device`.

### 16.5 NaN/Inf loss

Triệu chứng trong log:

```text
[WARNING] Bắt gặp NaN loss tại epoch ..., step ...
```

Nguyên nhân:

- fp16 có dải số hẹp.
- Attention mask dùng giá trị âm lớn.
- Custom bottleneck + QLoRA có thể gây bất ổn ở một vài sample.

Khắc phục:

- GradScaler.
- Gradient clipping `0.5`.
- AdamW `eps=1e-6`.
- Trainable params fp32.
- Skip batch NaN để không phá optimizer state.

Điểm cần nói khi thầy hỏi:

> NaN không làm hỏng toàn bộ training vì code phát hiện loss NaN trước backward/optimizer step, bỏ batch đó, zero grad và tiếp tục. Loss trung bình vẫn giảm qua epoch, chứng minh training còn ổn định.

### 16.6 Lỗi GradScaler với FP16 gradients

Triệu chứng:

```text
ValueError: Attempting to unscale FP16 gradients.
```

Nguyên nhân:

- Một số trainable params ở fp16.
- GradScaler không cho unscale FP16 gradients theo cách đó.

Khắc phục:

- Cast trainable params về fp32 trong `setup_peft_and_freezing`.

### 16.7 Lỗi `mrope_position_ids`/RoPE khi generate

Nguyên nhân:

- Version `transformers` khác nhau hỗ trợ tên tham số khác nhau.
- Qwen2.5-VL cần position ids multi-modal đúng.

Khắc phục:

- Dùng `inspect.signature` để kiểm tra forward hỗ trợ tham số nào.
- Tính `position_ids` trước khi thay attention mask.
- Không hardcode truyền tham số không tương thích.

### 16.8 Lỗi attention mask không contiguous khi inference Stage 1

Triệu chứng:

```text
RuntimeError: (*bias): last dimension must be contiguous
```

Nguyên nhân:

- SDPA backend yêu cầu bias/attention mask có layout contiguous.
- Custom 4D mask trong generate/inference dễ có stride không phù hợp.

Khắc phục hướng đúng:

- Gọi `.contiguous()` sau khi stack/cast mask.
- Hoặc tránh dùng Stage 1 generate sanity nếu chưa patch inference mask chắc chắn.
- Trong evaluation cuối, ưu tiên Stage 2 chính thức và dùng Stage 1 sanity cẩn trọng.

## 17. Bằng chứng training đã chạy

Output cũ trong `livr-mini-benchmark.ipynb` cho thấy:

```text
Không phát hiện checkpoint Stage 1. Sẽ huấn luyện từ Epoch 1...
Epoch 1/5 (Stage 1): 1200/1200
Epoch 2/5 (Stage 1): 1200/1200
Epoch 3/5 (Stage 2): 1200/1200
Epoch 4/5 (Stage 2): 1200/1200
Epoch 5/5 (Stage 2): 1200/1200
```

Loss trung bình:

```text
Epoch 1: 0.3319
Epoch 2: 0.2663
Epoch 3: 0.1814
Epoch 4: 0.1530
Epoch 5: 0.1036
```

Kết luận:

- Training bắt đầu từ epoch 1, không dùng checkpoint Stage 1.
- Đã train nhiều hơn yêu cầu tối thiểu 1 epoch.
- Loss giảm đều, chứng minh mô hình học được tín hiệu.
- Final checkpoint được lưu tại `/kaggle/working/checkpoints/livr_mini_checkpoint.pt`.

## 18. Những điểm nên nói rõ khi seminar

### 18.1 "From scratch" không phải train Qwen từ random init

Câu trả lời đề xuất:

> Trong điều kiện mini project, from scratch nghĩa là bắt đầu từ base pretrained Qwen2.5-VL và không dùng checkpoint LIVR đã huấn luyện trước. Em train mới toàn bộ phần adapter và latent embeddings. Train full Qwen từ random initialization là một bài toán pretraining hàng tỷ tham số, không phù hợp tài nguyên Kaggle.

### 18.2 "Tại sao dùng QLoRA?"

> Vì Qwen2.5-VL-3B nếu fine-tune full hoặc load fp16 sẽ vượt VRAM khi có ảnh. QLoRA giữ backbone ở 4-bit, freeze nó, chỉ train LoRA adapters và latent embeddings. Cách này giảm VRAM nhưng vẫn cho phép mô hình thích nghi với dữ liệu LIVR.

### 18.3 "Tại sao thêm latent tokens?"

> Latent tokens là bộ nhớ ẩn để ép mô hình học biểu diễn thị giác ngầm. Trong Stage 1, answer không được nhìn trực tiếp image tokens, nên nếu muốn trả lời đúng, thông tin ảnh phải được nén vào latent tokens.

### 18.4 "Tại sao phải có Stage 2?"

> Stage 1 dạy latent tokens chứa thông tin ảnh. Stage 2 mở lại attention thường để mô hình phối hợp latent representation với visual tokens gốc. Nếu chỉ Stage 1, mô hình có thể mất chi tiết; nếu chỉ Stage 2, latent tokens dễ bị bỏ qua.

### 18.5 "Tại sao chỉ train 4 task?"

> Vì đây là bản mini trên Kaggle/Colab. Dataset gốc có nhiều task và ảnh lớn. Em chọn 4 task thị giác cốt lõi để kiểm chứng cơ chế LIVR: counting, localization, jigsaw, visual similarity. Các task này đủ đại diện cho reasoning thị giác mà vẫn khả thi với tài nguyên hạn chế.

### 18.6 "Kết quả sanity 20 mẫu có phải benchmark không?"

> Không. Cell cuối của implement chỉ là sanity check nhanh trên 20 mẫu để xác nhận model sau train tốt hơn base khi disable adapter. Benchmark chính thức nằm ở notebook evaluation, với split và log riêng.

## 19. Câu hỏi dự phòng thầy có thể hỏi

### Hỏi: Vì sao adapter LoRA gắn vào cả attention và MLP?

Trả lời:

> Attention quyết định luồng thông tin giữa ảnh, prompt, latent và answer. MLP giúp biến đổi phi tuyến biểu diễn sau attention. Gắn cả hai giúp mô hình vừa học cách attend đúng, vừa học cách chuyển thông tin latent thành phân phối answer tốt hơn.

### Hỏi: Vì sao không update vision encoder?

Trả lời:

> Vision encoder của Qwen đã được pretrained mạnh. Update nó sẽ tốn VRAM lớn và dễ overfit với tập mini. Mục tiêu của em là kiểm chứng cơ chế latent bottleneck, nên em giữ encoder ổn định và chỉ học adapter ở language transformer cùng latent embeddings.

### Hỏi: Latent tokens có được supervise trực tiếp không?

Trả lời:

> Không. Latent tokens học gián tiếp qua answer loss. Đây chính là điểm hay của LIVR: không cần label trung gian như bounding box, depth map hay chain-of-thought. Bottleneck mask tạo áp lực để latent tokens phải chứa thông tin ảnh.

### Hỏi: Nếu answer bị chặn nhìn ảnh trong Stage 1, mô hình lấy thông tin ảnh bằng cách nào?

Trả lời:

> Latent tokens vẫn được phép attend vào image tokens. Answer tokens được phép attend vào latent tokens theo causal attention. Nếu nói theo luồng thông tin thì là Image -> Latent -> Answer; nếu nói đúng theo hướng attention thì là Answer attend Latent attend Image.

### Hỏi: Vì sao loss vẫn giảm dù có batch NaN?

Trả lời:

> Vì code kiểm tra NaN trước backward update. Batch lỗi bị bỏ qua, optimizer không step trên gradient hỏng. Các batch còn lại vẫn cung cấp tín hiệu học. Loss trung bình giảm từ 0.3319 xuống 0.1036 là bằng chứng training vẫn ổn.

### Hỏi: Có data leakage không?

Trả lời:

> Phần implement train trên `livr_train`, không phải benchmark test chính thức. Cell 20 mẫu cuối chỉ là sanity check trên train samples nên không được xem là benchmark. Evaluation chính thức cần tách train/test hoặc dùng dataset external như CV-Bench/MathVista như notebook evaluation đã làm.

### Hỏi: Vì sao checkpoint nhỏ?

Trả lời:

> Vì em chỉ lưu phần trainable delta: LoRA weights và latent embeddings. Backbone Qwen không đổi và có thể load lại từ Hugging Face. Đây là cách lưu đúng cho PEFT/QLoRA.

### Hỏi: Nếu dùng A100 thì có cần QLoRA không?

Trả lời:

> Không bắt buộc. Với A100 có thể dùng bf16 LoRA hoặc full fine-tune một phần lớn hơn. Nhưng QLoRA vẫn hữu ích vì tiết kiệm VRAM, cho phép tăng batch size hoặc tăng số mẫu/epoch.

### Hỏi: Vì sao dùng fp16 thay vì bf16?

Trả lời:

> GPU T4 không hỗ trợ bf16 tốt như A100/L4. fp16 tận dụng Tensor Cores của T4 tốt hơn. Vì fp16 dễ NaN hơn, em bổ sung GradScaler, clipping, eps AdamW và fp32 trainable params.

### Hỏi: Vì sao dùng `-10000.0` trong mask thay vì `-inf`?

Trả lời:

> Trong fp16, dùng `-inf` hoặc giá trị quá âm dễ gây NaN khi cộng attention scores và softmax. Một số âm hữu hạn đủ lớn như `-10000.0` vẫn làm xác suất sau softmax gần bằng 0, nhưng ổn định số học hơn.

### Hỏi: Tại sao phải tính position ids trước khi thay attention mask?

Trả lời:

> Qwen2.5-VL cần M-RoPE position ids cho text và image tokens. Nếu thay attention mask 2D bằng custom mask 4D quá sớm, hàm nội bộ có thể không suy ra đúng position ids. Vì vậy patch tính RoPE bằng mask gốc trước, rồi mới thay bằng bottleneck mask.

### Hỏi: Điểm yếu lớn nhất của implement này là gì?

Trả lời:

> Có ba điểm: quy mô mini nên số task/samples ít hơn bài báo; monkey-patch phụ thuộc version transformers; và sanity evaluation trong notebook implement không phải benchmark chính thức. Em đã xử lý bằng cách ghi rõ phạm vi mini, pin transformers trong requirements, và tách notebook evaluation chính thức.


## 20. Giải thích sâu các file `.py` khi bị hỏi code

Phần này dùng để trả lời các câu hỏi kiểu: "đoạn này trong code là gì?", "vì sao phải monkey-patch?", "RoPE là gì?", "tại sao phải xử lý `position_ids`?". Đây là nhóm câu hỏi rất dễ xuất hiện khi thầy mở trực tiếp file `.py`.

### 20.1 Monkey-patch là gì?

**Monkey-patch không phải là tên một hàm cụ thể**, mà là một kỹ thuật trong Python: thay đổi hoặc bọc lại hành vi của một object/class/module ngay trong lúc chương trình đang chạy, không cần sửa source code gốc của thư viện.

Trong dự án này, hàm thực hiện kỹ thuật đó là `patch_model_for_livr` trong [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py:37).

Code cốt lõi:

```python
original_forward = model.forward
...
model.forward = types.MethodType(livr_forward, model)
```

Ý nghĩa:

1. Lưu lại `forward` gốc của Qwen vào `original_forward`.
2. Viết một hàm mới `livr_forward` để can thiệp trước khi gọi model.
3. Trong `livr_forward`, nếu đang ở Stage 1 thì tạo bottleneck attention mask.
4. Sau khi chỉnh `attention_mask`/`position_ids`, gọi lại `original_forward` để Qwen chạy bình thường.
5. Gán `model.forward` thành hàm mới bằng `types.MethodType`.

Nói đơn giản:

> Monkey-patch ở đây là bọc quanh hàm `forward` gốc của Qwen để chèn attention mask LIVR vào đúng thời điểm, nhưng vẫn giữ toàn bộ logic pretrained của Qwen.

Nếu thầy hỏi "tại sao không sửa source code Qwen luôn?", trả lời:

> Vì Qwen2.5-VL nằm trong thư viện `transformers`. Sửa trực tiếp source thư viện sẽ khó tái lập, dễ lỗi khi Kaggle cài version khác, và không sạch về mặt engineering. Monkey-patch giúp em can thiệp tối thiểu vào runtime, giữ nguyên model gốc, nhưng vẫn thêm được cơ chế bottleneck mask của LIVR.

Nếu thầy hỏi "tại sao không subclass model?", trả lời:

> Subclass Qwen2.5-VL khi đã có PEFT, LoRA, `device_map="auto"`, Accelerate hooks và quantization sẽ phức tạp hơn. Monkey-patch là cách ít xâm lấn hơn: chỉ bọc `forward`, còn loading, LoRA, quantization và device placement vẫn do Hugging Face/PEFT xử lý.

Rủi ro của monkey-patch:

- Phụ thuộc vào signature `forward` của phiên bản `transformers`.
- Khi Hugging Face đổi tên tham số, patch có thể lỗi.
- Khó debug hơn gọi API chuẩn.

Cách dự án giảm rủi ro:

- Dùng `inspect.signature` để kiểm tra forward hỗ trợ tham số nào.
- Giữ `original_forward` để luôn gọi lại logic gốc.
- Chỉ can thiệp Stage 1 training khi có `labels`, còn Stage 2 dùng forward gốc.

### 20.2 `types.MethodType` là gì?

Trong [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py:194):

```python
model.forward = types.MethodType(livr_forward, model)
```

`types.MethodType` biến một function bình thường thành method gắn với object `model`. Khi gọi:

```python
model(**inputs)
```

Python sẽ tự truyền `self=model` vào `livr_forward`.

Nếu không dùng `MethodType`, hàm mới có thể không nhận đúng `self`, hoặc hoạt động như function rời thay vì method của object.

### 20.3 `inspect.signature` dùng để làm gì?

Trong [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py:41), code lấy signature của `original_forward`:

```python
original_forward_sig = inspect.signature(original_forward)
original_forward_params = set(original_forward_sig.parameters)
```

Mục đích là kiểm tra phiên bản model hiện tại nhận những tham số nào. Ví dụ có version nhận `rope_deltas`, có version khác lại không nhận, có version cũ từng liên quan `mrope_position_ids`.

Nếu truyền một keyword mà forward không hỗ trợ, sẽ lỗi kiểu:

```text
TypeError: forward() got an unexpected keyword argument ...
```

Vì vậy code có hàm:

```python
def forward_supports(name):
    return original_forward_accepts_kwargs or name in original_forward_params
```

Đây là cách làm patch tương thích nhiều version `transformers` hơn.

### 20.4 `*args` và `**kwargs` trong `livr_forward`

Trong [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py:62), hàm wrapper nhận:

```python
def livr_forward(self, *args, **kwargs):
```

Lý do: Hugging Face/PEFT có thể gọi model bằng positional arguments hoặc keyword arguments. Nếu chỉ viết cứng `input_ids=...`, `attention_mask=...`, patch dễ hỏng.

Hai helper:

```python
def get_arg(name, index, default=None): ...
def set_arg(name, index, value): ...
```

cho phép lấy và ghi lại tham số dù nó nằm trong `kwargs` hay trong `args`.

Nếu thầy hỏi "tại sao code vòng vèo vậy?", trả lời:

> Vì model có thể được gọi qua nhiều lớp wrapper PEFT/Accelerate, tham số có thể đi vào dưới dạng positional hoặc keyword. `get_arg`/`set_arg` giúp patch không phụ thuộc cứng vào một kiểu gọi duy nhất.

### 20.5 RoPE là gì?

**RoPE** là viết tắt của **Rotary Position Embedding**. Đây là kỹ thuật mã hóa vị trí token bằng cách xoay vector Query/Key trong attention theo vị trí của token.

Trong Transformer, attention tự nó không biết token nào đứng trước token nào. Vì vậy model cần positional encoding. RoPE làm việc đó bằng cách "xoay" các chiều của vector theo vị trí:

```text
Token ở vị trí 0: góc xoay nhỏ/ban đầu
Token ở vị trí 10: góc xoay khác
Token ở vị trí 100: góc xoay khác nữa
```

Nhờ vậy, khi Query và Key tương tác trong attention, model biết được quan hệ vị trí tương đối giữa các token.

Câu trả lời ngắn khi bị hỏi:

> RoPE là cơ chế mã hóa vị trí bằng phép xoay vector Query/Key trong attention. Nó giúp transformer biết thứ tự và khoảng cách tương đối giữa các token mà không cần cộng positional embedding kiểu cũ.

### 20.6 M-RoPE trong Qwen2.5-VL là gì?

Qwen2.5-VL không chỉ xử lý text, mà còn xử lý ảnh. Với ảnh, vị trí không chỉ là thứ tự 1D như câu văn, mà còn có cấu trúc không gian 2D:

- chiều cao,
- chiều rộng,
- đôi khi có chiều thời gian nếu là video.

Vì vậy Qwen2.5-VL dùng cơ chế position embedding đa phương thức, thường gọi là **M-RoPE** hoặc multi-modal RoPE.

Trong code, các thông tin liên quan gồm:

- `image_grid_thw`: kích thước lưới ảnh/video theo temporal-height-width.
- `get_rope_index`: hàm nội bộ của Qwen để tính `position_ids` đúng cho text + image.
- `position_ids`: tensor vị trí đưa vào attention.
- `rope_deltas`: độ lệch vị trí mà Qwen cần để generate tiếp.

Nếu thầy hỏi "vì sao text model cần position ids thì hiểu rồi, nhưng VLM cần gì thêm?", trả lời:

> Vì ảnh sau processor được chia thành visual tokens theo lưới không gian. Model cần biết visual token nào nằm ở vùng nào trong ảnh, không chỉ biết nó là token thứ mấy trong chuỗi. M-RoPE giúp Qwen2.5-VL mã hóa vị trí đa phương thức của image tokens và text tokens cùng lúc.

### 20.7 Vì sao phải tính `position_ids` trước khi thay attention mask?

Trong [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py:152), comment ghi rõ:

```python
# Tự động tính toán position_ids bằng 2D attention_mask gốc trước khi ghi đè mask 4D
```

Bình thường Qwen nhận `attention_mask` 2D dạng:

```text
[batch, seq_len]
```

Từ đó Qwen tự tính vị trí/RoPE cho sequence. Nhưng LIVR Stage 1 cần thay attention mask bằng custom mask 4D dạng gần như:

```text
[batch, 1, seq_len, seq_len]
```

hoặc một biến thể tương thích với backend attention. Nếu đưa mask 4D vào quá sớm, hàm `get_rope_index` có thể không hiểu mask này theo cách Qwen mong đợi.

Vì vậy thứ tự đúng là:

1. Giữ `attention_mask` gốc để tính `position_ids`/RoPE.
2. Sau khi có `position_ids`, mới thay `attention_mask` bằng bottleneck mask.
3. Gọi `original_forward`.

Nếu thầy hỏi "không tính trước thì sao?", trả lời:

> Có thể lỗi shape, lỗi `mrope_position_ids`, hoặc model dùng sai vị trí cho image/text tokens. Với VLM, sai position ids làm attention hiểu sai cấu trúc ảnh và sequence.

### 20.8 `get_backbone_with_rope` để làm gì?

Trong [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py:52), code đi qua các lớp `.model`:

```python
def get_backbone_with_rope(root):
    backbone = root
    visited = set()
    while hasattr(backbone, "model") and id(backbone) not in visited:
        if hasattr(backbone, "get_rope_index"):
            return backbone
        visited.add(id(backbone))
        backbone = backbone.model
```

Lý do: sau khi dùng PEFT/LoRA, object model bị bọc nhiều lớp. Có thể `get_rope_index` không nằm ngay ở object ngoài cùng, mà nằm sâu bên trong backbone Qwen.

Hàm này đi xuống từng lớp `.model` để tìm object nào có `get_rope_index`.

Nếu thầy hỏi "tại sao không gọi thẳng `model.get_rope_index`?", trả lời:

> Vì sau khi qua PEFT, object ngoài cùng có thể là wrapper chứ không phải Qwen backbone thật. Gọi thẳng có thể không thấy method. Hàm này tìm đúng lớp bên trong có chức năng tính RoPE.

### 20.9 `attention_mask` trong code là mask cộng, không chỉ mask 0/1

Trong `generate_stage1_bottleneck_mask`, ban đầu code tạo boolean mask:

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
```

`torch.tril` tạo causal mask tam giác dưới: token hiện tại chỉ nhìn được token trước nó, không nhìn tương lai.

Sau đó các vị trí bị chặn được đặt `False`:

```python
mask[answer_indices.unsqueeze(1), img_indices] = False
mask[prompt_indices.unsqueeze(1), img_indices] = False
```

Cuối cùng đổi sang additive mask:

```python
float_mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
float_mask = float_mask.masked_fill(~mask, -10000.0)
```

Ý nghĩa:

- Vị trí được nhìn: cộng `0.0` vào attention score.
- Vị trí bị chặn: cộng `-10000.0` vào attention score.
- Sau softmax, score `-10000.0` gần như xác suất 0.

Câu trả lời ngắn:

> Mask của em là additive attention bias. Nó không xóa token, mà làm attention score ở vị trí bị cấm cực nhỏ để softmax gần như không chọn vị trí đó.

### 20.10 `image_pad_token_id` là gì?

Trong [src/model.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py:35):

```python
self.image_pad_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
```

Qwen2.5-VL biểu diễn ảnh trong chuỗi input bằng các token placeholder như `<|image_pad|>`. Khi processor xử lý ảnh, các vị trí này tương ứng với visual tokens.

Trong mask code, ta tìm vị trí image bằng:

```python
img_positions = torch.where(curr_ids == image_pad_token_id)[0]
```

Nhờ đó code biết đoạn nào trong `input_ids` là image tokens để chặn prompt/answer attend trực tiếp vào ảnh trong Stage 1.

Nếu thầy hỏi "làm sao em biết token nào là ảnh?", trả lời:

> Em dùng token id đặc biệt `<|image_pad|>` của tokenizer Qwen2.5-VL. Processor chèn token này vào sequence tại vị trí visual tokens, nên em dò các vị trí có id đó để xác định vùng ảnh.

### 20.11 `labels is not None` trong patch có ý nghĩa gì?

Trong [src/mask_kaggle.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/mask_kaggle.py:93):

```python
if stage == 1 and labels is not None and input_ids is not None:
```

Điều kiện này làm custom bottleneck mask chỉ áp dụng khi training có labels. Khi generate/evaluate không có labels, patch này không can thiệp.

Lý do:

- Training Stage 1 cần bottleneck mask để ép latent học.
- Generate với custom 4D mask phức tạp hơn vì có KV cache, position update từng token, và SDPA yêu cầu shape/layout chặt.
- Tránh làm hỏng generation thông thường trong implement notebook.

Nếu muốn Stage 1 sanity trong generation, cần patch inference riêng và xử lý contiguous mask/KV cache kỹ hơn. Đây chính là lý do trước đó evaluation Stage 1 sanity từng gặp lỗi mask/SDPA.

### 20.12 `AutoProcessor` là gì?

Trong [src/model.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py:24), ta dùng:

```python
AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
```

`AutoProcessor` của Qwen2.5-VL bao gồm:

- tokenizer cho text,
- image processor cho ảnh,
- chat template formatter,
- logic biến ảnh thành `pixel_values` và `image_grid_thw`.

Nếu chỉ dùng tokenizer, model sẽ không nhận được ảnh đúng format. Với VLM, processor quan trọng ngang tokenizer.

### 20.13 `min_pixels` và `max_pixels` là gì?

Trong [src/model.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py:22):

```python
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
```

Qwen2.5-VL chia ảnh thành patch/grid. `max_pixels` giới hạn độ phân giải ảnh sau xử lý, từ đó giới hạn số visual tokens.

Nếu tăng `max_pixels`:

- ảnh chi tiết hơn,
- nhưng visual tokens nhiều hơn,
- attention tốn VRAM hơn,
- dễ OOM hơn.

Nếu giảm quá thấp:

- tiết kiệm VRAM,
- nhưng mất chi tiết ảnh,
- accuracy có thể giảm.

### 20.14 `resize_token_embeddings` để làm gì?

Sau khi thêm latent tokens vào tokenizer:

```python
self.processor.tokenizer.add_special_tokens(...)
```

Vocabulary size tăng lên. Nhưng embedding matrix của model vẫn có kích thước cũ. Vì vậy phải gọi:

```python
self.model.resize_token_embeddings(len(self.processor.tokenizer))
```

Nếu không gọi, khi input chứa `<latent_0>` đến `<latent_15>`, token id mới có thể vượt kích thước embedding matrix và gây lỗi index out of range.

### 20.15 `register_hook` là gì?

Trong [src/model.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py:105):

```python
embed_tokens.weight.register_hook(make_embedding_hook(latent_ids_tensor))
```

`register_hook` đăng ký một hàm chạy trong backward pass, khi gradient của tensor được tính xong.

Ở đây hook nhận gradient của toàn bộ embedding matrix, rồi nhân mask:

- latent rows giữ gradient,
- non-latent rows gradient bằng 0.

Câu trả lời ngắn:

> `register_hook` cho em can thiệp vào gradient sau khi backward tính xong nhưng trước khi optimizer cập nhật. Em dùng nó để chỉ cho phép 16 latent token embeddings học, còn vocabulary gốc không bị đổi.

### 20.16 `enable_input_require_grads` để làm gì?

Trong [src/model.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py:86):

```python
self.model.enable_input_require_grads()
```

Khi dùng gradient checkpointing + PEFT, đôi khi input embeddings cần `requires_grad` để backward đi qua đúng đường adapter. Dòng này giúp đảm bảo gradient flow ổn định khi model bị freeze phần lớn tham số.

Nếu thầy hỏi, trả lời:

> Dòng này là cấu hình tương thích giữa gradient checkpointing và PEFT. Nó đảm bảo input embeddings tham gia graph đúng cách để gradient truyền tới LoRA/latent embeddings.

### 20.17 `device_map="auto"` là gì?

Trong [src/model.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py:51):

```python
device_map="auto"
```

Hugging Face Accelerate tự quyết định đặt layer nào lên GPU nào/thiết bị nào. Trên Kaggle có thể có T4 x2, nên `auto` giúp phân phối model hợp lý hơn.

Rủi ro:

- Một số tensor custom nếu tự tạo trên CPU sẽ lệch device.

Do đó trong mask code phải luôn tạo tensor với `device=input_ids.device`.

### 20.18 `attn_implementation="sdpa"` là gì?

`sdpa` là Scaled Dot Product Attention backend của PyTorch. Nó dùng API tối ưu của PyTorch cho attention.

Lợi ích:

- nhanh hơn attention Python thuần,
- tiết kiệm bộ nhớ hơn,
- phù hợp GPU hiện đại.

Nhưng SDPA cũng nhạy với shape/layout của attention mask, nên Stage 1 inference từng có lỗi kiểu:

```text
(*bias): last dimension must be contiguous
```

Khi bị hỏi, trả lời:

> Em dùng SDPA để tối ưu tốc độ và VRAM. Khi truyền custom mask, phải chú ý tensor layout như contiguous, vì backend attention tối ưu yêu cầu shape/stride chặt hơn.

### 20.19 `src/utils.py` xử lý dataset như thế nào?

File [src/utils.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/utils.py:9) có ba vai trò chính.

Thứ nhất, tải dataset:

```python
snapshot_download(... allow_patterns=["train/**"])
```

Mục đích là tải đúng thư mục train về SSD, tránh lazy-loading ảnh từng file.

Thứ hai, lọc và format dữ liệu:

```python
filter_and_deduplicate_pipeline(...)
```

Code lọc task theo `dataset_name`, lấy `ground_truth` làm answer, ghép `choices` vào question nếu có, và tạo conversation theo Qwen chat format.

Lưu ý kỹ thuật:

- Code hiện tại dùng `seen_hashes` để loại ảnh có pHash trùng chính xác.
- Nếu muốn loại near-duplicate theo Hamming distance, cần mở rộng thêm so sánh khoảng cách hash. Không nên nói quá rằng code hiện tại đã dùng threshold nếu chưa implement threshold.

Thứ ba, chuẩn bị input:

```python
prepare_vqa_inputs(...)
```

Hàm này chèn latent tokens, áp chat template, load ảnh, gọi processor, đưa tensor lên GPU và tạo labels.

### 20.20 Vì sao `images=[images]` mà không phải `images=images`?

Trong [src/utils.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/utils.py:216):

```python
full_inputs = processor(text=[full_text], images=[images], padding=True, return_tensors="pt")
```

`text=[full_text]` nghĩa là batch size bằng 1. Với mỗi sample, Qwen processor mong `images` là danh sách ảnh tương ứng với từng text. Vì `images` bên trong đã là danh sách ảnh của một conversation, nên cần bọc thêm một lớp:

```text
images = [[image_1, image_2, ...]]
```

Trong dự án mỗi sample thường có 1 ảnh, nhưng cấu trúc vẫn là list trong list để đúng batch format.

### 20.21 `apply_chat_template` có vai trò gì?

Trong [src/utils.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/utils.py:196):

```python
full_text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=not is_training)
```

Qwen Instruct không nhận text thô đơn giản. Nó cần format hội thoại gồm role user/assistant, special tokens, generation prompt. `apply_chat_template` tạo đúng format mà model đã được instruction-tuned.

Nếu không dùng chat template, model có thể hiểu sai đâu là câu hỏi, đâu là câu trả lời.

### 20.22 `labels[:, :prompt_len] = -100` nghĩa là gì?

Trong [src/utils.py](/home/ductien/Documents/LIVR-Mini-Benchmark/src/utils.py:224):

```python
labels[:, :prompt_len] = -100
```

`-100` là ignore index của loss. Các token trước answer không bị tính loss.

Ý nghĩa:

- Không bắt model học sinh lại câu hỏi.
- Không tính loss trên image placeholder hoặc latent tokens.
- Chỉ phạt model khi sinh answer sai.

Đây là đúng với supervised fine-tuning dạng causal LM cho VQA.

### 20.23 Cách trả lời tổng quát khi thầy mở code `.py`

Một câu trả lời có cấu trúc tốt:

> Dạ code của em chia làm ba tầng. `model.py` dựng backbone Qwen2.5-VL theo QLoRA, thêm latent tokens và kiểm soát tham số nào được học. `utils.py` xử lý dataset thành conversation đúng chat template và tạo labels chỉ tính loss trên answer. `mask_kaggle.py` là phần can thiệp LIVR: monkey-patch forward để ở Stage 1 thay attention mask bằng bottleneck mask, đồng thời xử lý RoPE/position ids để Qwen2.5-VL vẫn hiểu đúng vị trí text và image tokens.

## 21. Kết luận bảo vệ phần implement

Phần implement mini đã đạt các yêu cầu cốt lõi:

- Có training thật trên dataset `Kkuntal990/LIVR_mixed`.
- Không dùng checkpoint LIVR có sẵn trong lần chạy nghiệm thu.
- Train nhiều hơn 1 epoch.
- Có QLoRA để chạy được trên Kaggle/Colab.
- Có latent token expansion đúng tinh thần LIVR.
- Có two-stage training với bottleneck mask.
- Có checkpoint cuối và loss curve giảm.
- Có xử lý các lỗi thực tế: OOM, NaN, GradScaler FP16, HF rate limit, checkpoint rỗng, RoPE/mask compatibility.

Cách trình bày nên trung thực:

> Đây là bản tái hiện mini có kiểm soát, không phải reproduction full-scale của bài báo. Điểm chính của project là chứng minh em hiểu và implement được cơ chế LIVR trong điều kiện tài nguyên hạn chế: QLoRA + latent tokens + bottleneck attention + two-stage training + evaluation tách riêng.
