# Báo Cáo Chi Tiết Evaluation LIVR-Mini-Benchmark Trên Kaggle

## 1. Mục Tiêu Của Evaluation

Mục tiêu của notebook `evaluation-add-task-prefix-and-helper.ipynb` / `Evaluation_03.ipynb` là xây dựng một phép đánh giá có kiểm soát cho cơ chế **LIVR - Latent Implicit Visual Reasoning** trong điều kiện tài nguyên hạn chế của Kaggle T4.

Đây **không phải** là reproduction đầy đủ của paper LIVR. Paper gốc đánh giá trên nhiều perception-heavy tasks, nhiều backbone, và setup huấn luyện lớn hơn. Phiên bản này là **mini-benchmark** để trả lời câu hỏi hẹp hơn:

> Khi dùng cùng một base model, cùng tập train/test, cùng ngân sách epoch, liệu LIVR có đem lại lợi ích so với Direct SFT thông thường hay không?

Vì vậy, trong báo cáo nên gọi đây là **controlled Kaggle-scale evaluation inspired by LIVR**, không nên gọi là full paper reproduction.

## 2. Liên Hệ Với Paper LIVR

Paper LIVR đề xuất một cơ chế giúp Large Multimodal Model học các biểu diễn thị giác ẩn thông qua **latent tokens**.

Các thành phần cốt lõi của LIVR trong paper:

- Thêm `K` latent tokens vào prompt. Trong project này dùng `K = 16`.
- Stage 1 dùng **visual bottleneck attention mask**: answer tokens không được nhìn trực tiếp image tokens; thông tin ảnh phải đi qua latent tokens.
- Stage 2 quay về standard causal/full-vision mask: model học kết hợp ảnh gốc và latent tokens để trả lời.
- So sánh với **Direct SFT** trên cùng task data và cùng training setup.

Trong mini-benchmark này, chúng ta giữ đúng tinh thần trên:

| Thành phần | Paper LIVR | Mini Evaluation |
| :--- | :--- | :--- |
| Latent tokens | Có | Có, `K=16` |
| Stage 1 bottleneck | Có | Có |
| Stage 2 standard mask | Có | Có |
| Direct SFT baseline | Có | Có |
| Data-matched comparison | Có | Có |
| Multi-backbone / many tasks | Có | Không, do giới hạn Kaggle |
| Full-scale training | Có | Không, chỉ 500 train + 200 test mỗi dataset |

## 3. Dataset Và Lý Do Lựa Chọn

Notebook dùng hai dataset có ảnh inline trên Hugging Face, phù hợp với Kaggle và không cần tải repo ảnh ngoài quá lớn.

### 3.1. CV-Bench

CV-Bench tập trung vào các tác vụ visual reasoning/perception-heavy:

- 2D counting/relation.
- 3D depth/distance.
- Object-centric/spatial reasoning.

Schema quan trọng:

- `prompt`: câu hỏi đã gồm choices.
- `choices`: danh sách lựa chọn.
- `answer`: option label dạng `(A)`, `(B)`, ...
- `type`, `source`, `task`: metadata dùng để stratified split.

CV-Bench phù hợp với LIVR hơn vì nó gần với các tác vụ perception-heavy mà paper LIVR hướng đến.

### 3.2. MathVista

MathVista là visual mathematical reasoning benchmark, gồm:

- Geometry.
- Chart/table reasoning.
- Counting/arithmetic.
- Multiple-choice và free-form answer.

Schema quan trọng:

- `query`: prompt chính thức của dataset, có hint về kiểu đáp án.
- `question`: câu hỏi gốc, nhưng không đầy đủ bằng `query`.
- `choices`: có nếu là multiple-choice.
- `answer`: với multiple-choice, thường là **text option**, không phải option letter.
- `question_type`, `answer_type`: metadata dùng để stratified split.

Điểm quan trọng: dùng `query` **không phải leakage**, vì đây là input chính thức của dataset và không chứa đáp án đúng. Hint như "provide final value" hoặc "provide correct option letter" chỉ chuẩn hóa format trả lời.

## 4. Protocol Evaluation

Protocol cuối cùng:

| Tham số | Giá trị |
| :--- | :--- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Training method | QLoRA 4-bit |
| Latent tokens | `K=16` |
| Train samples | 500 / dataset |
| Test samples | 200 / dataset |
| Combined train | 500 MathVista + 500 CV-Bench = 1000 samples |
| Direct SFT epochs | 5 |
| LIVR Stage 1 epochs | 2 |
| LIVR Stage 2 epochs | 3 |
| Total LIVR epochs | 5 |
| Metric | Exact/structured answer accuracy |

Ba mốc so sánh:

| Method | Train | Latent tokens | Bottleneck mask | Vai trò |
| :--- | :--- | :---: | :---: | :--- |
| Zero-shot | Không | Không | Không | Năng lực base model |
| Direct SFT | 5 epoch | Không | Không | Baseline fine-tuning chuẩn |
| LIVR | 2 epoch Stage 1 + 3 epoch Stage 2 | Có | Có ở Stage 1 | Phương pháp cần kiểm chứng |

Điều quan trọng nhất về fairness:

- Direct SFT và LIVR đều xuất phát từ cùng một base model.
- Notebook snapshot LoRA state ban đầu và latent embeddings ban đầu.
- Trước Direct SFT và trước LIVR, model được restore về cùng snapshot.
- Không load checkpoint Direct SFT bên ngoài.
- Không zero LoRA thủ công.
- Không dùng fallback hardcode kết quả.
- Zero-shot, Direct SFT, LIVR dùng cùng prompt format và cùng evaluation loop.

## 5. Split Strategy: Vì Sao Không Lấy 500 Đầu / 200 Cuối

Ban đầu có ý tưởng lấy 500 mẫu đầu làm train và 200 mẫu cuối làm test. Tuy nhiên, khi kiểm tra schema CV-Bench, dataset được sắp theo cụm source/task.

Nếu lấy tuần tự:

- 500 mẫu đầu CV-Bench chủ yếu là `2D/ADE20K`.
- 200 mẫu cuối CV-Bench chủ yếu là `3D/Omni3D/Distance`.

Điều này làm train/test bị lệch domain nặng, không còn công bằng cho mini evaluation.

Vì vậy notebook đổi sang **deterministic stratified split**:

- CV-Bench stratify theo `type/source/task`.
- MathVista stratify theo `question_type/answer_type`.
- Seed cố định `42` để có thể lặp lại.

Split audit trong run:

```text
CV-Bench train: n=500
  3D|Omni3D|Depth: 114
  3D|Omni3D|Distance: 114
  2D|COCO|Count: 84
  2D|COCO|Relation: 68
  2D|ADE20K|Count: 65
  2D|ADE20K|Relation: 55

CV-Bench test: n=200
  3D|Omni3D|Depth: 46
  3D|Omni3D|Distance: 45
  2D|COCO|Count: 34
  2D|COCO|Relation: 27
  2D|ADE20K|Count: 26
  2D|ADE20K|Relation: 22

MathVista train: n=500
  multi_choice|text: 270
  free_form|integer: 209
  free_form|float: 20
  free_form|list: 1

MathVista test: n=200
  multi_choice|text: 108
  free_form|integer: 84
  free_form|float: 8
```

## 6. Prompt Policy Và Answer Matching

### 6.1. Task Prefix

Một vấn đề ban đầu là model trả lời MathVista quá dài, ví dụ giải thích nhiều bước thay vì chỉ trả đáp án. Điều này gây rủi ro cho metric, vì câu trả lời dài có thể tình cờ chứa số đúng.

Notebook thêm task prefix:

- CV-Bench: yêu cầu trả lời option letter, không giải thích.
- MathVista: nếu multiple-choice thì trả lời option letter; nếu free-form thì trả lời final exact value hoặc phrase ngắn.

Task prefix được áp dụng đồng đều cho:

- Zero-shot.
- Direct SFT train/eval.
- LIVR train/eval.

Vì vậy đây là prompt-format policy chung, không phải trick riêng cho LIVR.

### 6.2. MathVista `query`

MathVista có `query` chứa hint chính thức và choices. Notebook ưu tiên:

- MathVista: `query` > `prompt` > `question`.
- CV-Bench: `prompt` > `query` > `question`.

Dùng `query` không phải leakage vì nó không chứa answer.

### 6.3. Answer Aliases Cho Multiple-Choice

MathVista multiple-choice thường có `answer` là text option, ví dụ `145°`, trong khi prompt yêu cầu trả lời option letter. Notebook tạo aliases để chấp nhận cả:

- `(C)`.
- `C`.
- `145°`.
- `(C) 145°`.
- `C. 145°`.

Điều này giúp chuyển đổi đúng giữa convention của dataset và convention của generated answer.

### 6.4. Một Bug Alias Nhỏ Cần Ghi Nhận

Khi kiểm tra log, có 1 sample MathVista có `original_ground_truth = "(b)"`, trong khi choices là:

```text
(A) (c)
(B) (d)
(C) (a)
(D) (b)
(E) (e)
```

Logic alias cũ có thể hiểu `"(b)"` thành option `(B)` thay vì text choice `(b)` nằm ở option `(D)`.

Khi tính lại bằng corrected alias logic, tổng accuracy của Zero-shot, Direct SFT, LIVR không thay đổi trong run này. Tuy nhiên, nên sửa logic này để sạch hơn trong các run sau: với MathVista, nếu `answer` là text option thì cần match exact text của choices trước, rồi mới fallback sang option-letter parsing.

## 7. Kỹ Thuật Vượt Giới Hạn Phần Cứng Kaggle

Kaggle T4 chỉ có khoảng 15GB VRAM, trong khi Qwen2.5-VL-3B là multimodal model lớn. Notebook dùng nhiều kỹ thuật để có thể train/evaluate được.

### 7.1. QLoRA 4-bit

Model được load với `BitsAndBytesConfig(load_in_4bit=True)`:

- Giảm VRAM của base model.
- Chỉ train LoRA adapters và latent embeddings.
- Base model gần như được freeze.

Cấu hình chính:

```python
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.float16
```

Dùng `float16` vì T4 không tối ưu cho `bfloat16`.

### 7.2. LoRA Target Modules

LoRA được gắn vào các module transformer quan trọng:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

Điều này giúp model học được cả attention và MLP transformations mà không cần fine-tune full model.

### 7.3. Freeze Vision Encoder Và Projector

Theo tinh thần LIVR và để tiết kiệm VRAM:

- Vision encoder/projector được freeze.
- Base LLM weights được freeze.
- Chỉ LoRA params và embedding rows của latent tokens được train.

### 7.4. Gradient Checkpointing

Notebook bật gradient checkpointing:

```python
gradient_checkpointing_enable(use_reentrant=False)
```

Lợi ích:

- Giảm VRAM đáng kể.
- Đổi lại training chậm hơn vì phải recompute activations.

Trong log có warning:

```text
use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...
```

Đây là warning bình thường. Khi train với gradient checkpointing, cache phải tắt.

### 7.5. AMP Autocast Và GradScaler

Forward pass dùng:

```python
torch.amp.autocast("cuda", dtype=torch.float16)
```

Để tránh gradient underflow, dùng:

```python
torch.amp.GradScaler("cuda")
```

`Scale` trong progress bar là hệ số loss scaling của AMP. Nếu gặp overflow/NaN, GradScaler sẽ giảm scale. Trong run, scale giảm nhưng training tiếp tục ổn định.

### 7.6. Trainable Params Ở Float32

Ban đầu gặp lỗi:

```text
ValueError: Attempting to unscale FP16 gradients.
```

Nguyên nhân: một số trainable params của LoRA/embedding ở FP16, trong khi GradScaler không cho unscale FP16 gradients.

Fix:

- Ép trainable params về `float32`.
- Thêm guard `ensure_trainable_params_fp32(model)`.

Log sau fix:

```text
Trainable dtype before/after guard: {'torch.float32': 347795456}; converted_tensors=0
```

### 7.7. Giới Hạn Kích Thước Ảnh

Processor được cấu hình `min_pixels` và `max_pixels` để tránh số visual tokens quá lớn:

```python
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
```

Điều này quan trọng với VLM vì ảnh lớn có thể tạo quá nhiều image tokens và gây OOM.

### 7.8. Gradient Accumulation

Vì batch size thực tế trên T4 rất nhỏ, notebook dùng gradient accumulation:

```text
grad_accumulation_steps = 8
```

Như vậy có thể mô phỏng effective batch lớn hơn mà không vượt VRAM.

### 7.9. OOM Guard Và NaN/Inf Guard

Training loop có:

- Catch CUDA OOM, dọn cache, skip batch.
- Check NaN/Inf loss, skip batch.
- Gradient clipping `max_norm=0.5`.

Trong run có một vài batch NaN/Inf bị skip, thường lặp lại ở một số sample 3D/Omni3D khó. Số batch bị skip rất ít so với tổng training steps, loss vẫn giảm đều, nên không làm vỡ kết quả.

### 7.10. Snapshot State Thay Vì Load 3 Model

Để tiết kiệm VRAM, notebook không load 3 bản model riêng cho Zero-shot, Direct SFT, LIVR.

Thay vào đó:

1. Load một model duy nhất.
2. Setup LoRA + latent tokens.
3. Snapshot LoRA tensors ban đầu và latent embedding rows.
4. Trước mỗi branch, restore về snapshot.

Điều này tiết kiệm rất nhiều VRAM và thời gian, nhưng vẫn giữ fairness giữa Direct SFT và LIVR.

### 7.11. Pin `transformers`

Notebook pin:

```text
transformers==4.51.3
```

Lý do: Kaggle mới có thể cài `transformers 5.0.0`, làm thay đổi API generation/Qwen2.5-VL, dẫn đến lỗi `prepare_inputs_for_generation` hoặc tham số không tương thích.

### 7.12. Patch Mask Cho Qwen2.5-VL

LIVR cần custom attention mask ở Stage 1. Qwen2.5-VL có M-RoPE/position ids riêng, nên khi thay attention mask 2D bằng mask 4D cần tính `position_ids` đúng trước khi forward.

Đã sửa các lỗi:

- Không truyền `mrope_position_ids` vào forward nếu version model không hỗ trợ.
- Chỉ can thiệp custom mask khi `model.livr_stage == 1`.
- Direct SFT và LIVR Stage 2 đi qua forward gốc để giảm rủi ro.

### 7.13. Patch Stage 1 Inference Sanity

Training Stage 1 có labels nên patch training forward kích hoạt. Nhưng khi `model.generate()`, không có labels, nên cần patch inference riêng để Stage 1 sanity vẫn dùng bottleneck mask.

Đã gặp lỗi:

```text
RuntimeError: (*bias): last dimension must be contiguous
```

Nguyên nhân: PyTorch SDPA yêu cầu attention bias/mask có last dimension contiguous.

Fix trong Kaggle session:

- Force `.contiguous()` cho attention mask.
- Nếu patch cũ còn bọc quanh forward, unwrap/patch lại.
- Có hotfix SDPA để ép `attn_mask.contiguous()` ngay trước scaled dot product attention.

Sau hotfix, Stage 1 sanity chạy thành công.

## 8. Kết Quả Cuối Cùng

### 8.1. Three-Way Comparison

| Dataset | Zero-shot | Direct SFT | LIVR Stage 2 | LIVR - SFT |
| :--- | ---: | ---: | ---: | ---: |
| CV-Bench | 68.5% | 80.5% | 86.0% | +5.5% |
| MathVista | 55.0% | 68.0% | 65.0% | -3.0% |
| Average | 61.75% | 74.25% | 75.50% | +1.25% |

### 8.2. Loss History

| Method | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Direct SFT | 0.2748 | 0.1485 | 0.0926 | 0.0748 | 0.0319 |
| LIVR | 0.4433 | 0.3372 | 0.1973 | 0.1225 | 0.0695 |

Diễn giải:

- Cả Direct SFT và LIVR đều học thật, loss giảm đều.
- LIVR loss cao hơn Direct SFT là hợp lý vì Stage 1 bị bottleneck nên bài toán khó hơn.
- Không còn vấn đề loss đứng im như ban đầu.

### 8.3. LIVR Stage 1 Sanity Check

| Dataset | LIVR Stage 2 | Stage 1 Sanity | Drop |
| :--- | ---: | ---: | ---: |
| CV-Bench | 86.0% | 68.5% | -17.5% |
| MathVista | 65.0% | 36.0% | -29.0% |

Diễn giải:

- Stage 1 sanity không sập về random hoàn toàn trên CV-Bench, nên latent tokens có học một phần thông tin thị giác.
- Tuy nhiên drop khá lớn, đặc biệt trên MathVista, cho thấy latent tokens chưa đủ khả năng giữ toàn bộ thông tin chi tiết khi bị cắt đường nhìn trực tiếp tới image tokens.

## 9. Phân Tích Theo Dataset

### 9.1. CV-Bench

Kết quả:

```text
Zero-shot: 68.5%
Direct SFT: 80.5%
LIVR Stage 2: 86.0%
LIVR - SFT: +5.5%
```

Đây là kết quả đẹp nhất của run.

Ý nghĩa:

- CV-Bench là spatial/perception-heavy benchmark, gần với mục tiêu của LIVR.
- LIVR giúp model cải thiện thêm so với Direct SFT khi cần suy luận thị giác về depth, distance, relation.
- Stage 1 sanity = 68.5% cho thấy latent bottleneck giữ được một phần thông tin thị giác, nhưng chưa bằng Stage 2.

Cách nói khi thuyết trình:

> Trên CV-Bench, LIVR vượt Direct SFT 5.5 điểm. Đây là bằng chứng tích cực nhất rằng latent bottleneck có lợi cho các tác vụ spatial/perception-heavy.

### 9.2. MathVista

Kết quả:

```text
Zero-shot: 55.0%
Direct SFT: 68.0%
LIVR Stage 2: 65.0%
LIVR - SFT: -3.0%
```

Nếu chỉ nhìn tổng số, LIVR thấp hơn Direct SFT 3 điểm. Khi soi chi tiết:

| Nhóm câu | Số mẫu | Zero-shot | Direct SFT | LIVR Stage 2 | LIVR - SFT |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Free-form | 92 | 30 | 52 | 54 | +2 |
| Multiple-choice | 108 | 80 | 84 | 76 | -8 |

Vậy LIVR không yếu trên mọi MathVista. Nó thực ra hơn Direct SFT nhẹ ở free-form/integer, nhưng thua ở multiple-choice.

Phân tích thêm theo nhóm coarse:

| Nhóm | n | Zero-shot | Direct SFT | LIVR Stage 2 | Stage 1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| chart/table | 14 | 8 | 12 | 10 | 5 |
| count/arithmetic | 30 | 14 | 24 | 25 | 7 |
| geometry | 36 | 16 | 19 | 18 | 15 |
| yes/no | 39 | 39 | 39 | 37 | 26 |
| other | 81 | 33 | 42 | 40 | 19 |

Nguyên nhân hợp lý:

- MathVista yêu cầu reasoning toán chi tiết, đọc biểu đồ, hình học, và phân biệt option tinh vi.
- Stage 1 bottleneck có thể làm latent tokens chưa học đủ các chi tiết cần thiết.
- Direct SFT học trực tiếp mapping prompt-answer của MathVista, nên mạnh hơn ở multiple-choice.
- LIVR có lợi ở một số câu free-form/counting, nhưng mất nhiều hơn ở multiple-choice.

Cách nói khi thuyết trình:

> MathVista cho thấy giới hạn của LIVR trong mini setting. LIVR không thắng Direct SFT trên tổng thể, chủ yếu do mất accuracy ở multiple-choice. Điều này gợi ý bottleneck 2 epoch chưa đủ để latent tokens lưu giữ thông tin toán học/option-level chi tiết.

## 10. Các Lỗi Đã Gặp Và Cách Khắc Phục

### 10.1. Transformers 5.0 / Generation API Mismatch

Vấn đề:

- Kaggle có thể cài `transformers 5.0.0`.
- Qwen2.5-VL generation API thay đổi.
- Gây lỗi trong generate/prepare inputs.

Fix:

- Pin `transformers==4.51.3`.

### 10.2. `mrope_position_ids` Unexpected Keyword

Vấn đề:

```text
Qwen2_5_VLForConditionalGeneration.forward() got an unexpected keyword argument 'mrope_position_ids'
```

Nguyên nhân:

- Patch mask truyền tham số không được version Qwen forward hỗ trợ.

Fix:

- Kiểm tra signature trước khi truyền.
- Nếu model hỗ trợ `rope_deltas` thì dùng `rope_deltas`.
- Nếu model hỗ trợ `mrope_position_ids` mới truyền.
- Nếu không, chỉ set `position_ids`.

### 10.3. FP16 GradScaler Error

Vấn đề:

```text
ValueError: Attempting to unscale FP16 gradients.
```

Nguyên nhân:

- Trainable LoRA/embedding params ở FP16.
- GradScaler không cho unscale FP16 gradients.

Fix:

- Ép trainable params về FP32.
- Thêm guard check trước optimizer step.

### 10.4. Stage 1 Inference SDPA Contiguous Error

Vấn đề:

```text
RuntimeError: (*bias): last dimension must be contiguous
```

Nguyên nhân:

- Custom 4D attention mask trong Stage 1 sanity generate không contiguous theo yêu cầu của PyTorch SDPA.

Fix:

- `.contiguous()` attention mask.
- Hotfix SDPA nếu patch cũ còn trong RAM.
- Repatch inference forward.

### 10.5. NaN/Inf Loss Ở Một Vài Batch

Run có một số warning:

```text
[WARNING] Direct SFT: NaN/Inf loss tại step ..., bỏ qua batch.
[WARNING] LIVR: NaN/Inf loss tại step ..., bỏ qua batch.
```

Phân tích:

- Các warning lặp ở một số sample khó, nhiều khả năng do FP16 numerical instability.
- Số batch bị skip rất nhỏ so với tổng training.
- Loss vẫn giảm đều.
- Evaluation vẫn hoàn tất.

Do đó đây là hạn chế kỹ thuật cần ghi nhận, nhưng không làm kết quả vô nghĩa.

## 11. Kết Luận Khoa Học

Kết quả ủng hộ một kết luận cân bằng:

1. LIVR có tín hiệu tích cực trên tác vụ spatial/perception-heavy.
   - CV-Bench: LIVR hơn Direct SFT `+5.5%`.

2. LIVR chưa ổn định trên visual mathematical reasoning.
   - MathVista: LIVR kém Direct SFT `-3.0%`.
   - Nguyên nhân chủ yếu nằm ở multiple-choice.

3. Trung bình hai dataset, LIVR nhẹ hơn Direct SFT.
   - Average LIVR - SFT = `+1.25%`.

4. Stage 1 sanity cho thấy latent tokens có học thông tin, nhưng bottleneck chưa đủ mạnh.
   - CV-Bench Stage 1: `68.5%`.
   - MathVista Stage 1: `36.0%`.

Câu kết luận nên dùng trong báo cáo:

> Trong mini Kaggle evaluation, LIVR cho thấy lợi ích rõ trên CV-Bench, một benchmark thiên về spatial/perception reasoning, nhưng chưa vượt Direct SFT trên MathVista, nơi cần suy luận toán học và phân biệt option chi tiết. Kết quả này ủng hộ giả thuyết latent bottleneck có ích cho một số dạng visual reasoning, đồng thời chỉ ra rằng cơ chế này cần thêm tuning/data/epoch để ổn định trên visual math.

## 12. Hạn Chế

Cần nói chủ động với thầy:

- Chỉ chạy một seed.
- Chỉ dùng một backbone Qwen2.5-VL-3B.
- Chỉ 500 train + 200 test mỗi dataset.
- Không phải official leaderboard protocol.
- MathVista `testmini` được dùng để domain adaptation mini, nên không so trực tiếp với leaderboard.
- Kaggle T4 giới hạn batch size, epoch, và precision.
- Stage 1 inference cần monkey patch riêng cho Qwen2.5-VL.
- Một vài NaN/Inf batch được skip.

## 13. Hướng Cải Thiện

Nếu có thêm thời gian/compute:

1. Chạy nhiều seed để đo variance.
2. Tăng train samples hoặc epochs cho MathVista.
3. Thử schedule LIVR khác, vì MathVista có thể cần Stage 1 dài hơn hoặc learning rate nhỏ hơn.
4. Tách train/eval theo dataset thay vì combined train để xem LIVR hợp domain nào.
5. Thêm validation split để early stopping.
6. Sửa canonical alias logic cho MathVista multiple-choice text như `(b)`.
7. Log metadata source/task vào eval details để phân tích chính xác hơn.
8. Thử LoRA rank/alpha khác.
9. Thử số latent tokens `K=8/16/32`.
10. Thử loss weighting riêng cho latent tokens hoặc auxiliary consistency loss.

## 14. Thầy Sẽ Muốn Nghe Gì Khi Thuyết Trình

Các câu hỏi khả năng cao:

### Vì sao cần Direct SFT?

Vì nếu chỉ so sánh LIVR với zero-shot, ta không biết cải thiện đến từ LIVR hay chỉ do fine-tuning. Direct SFT là baseline công bằng nhất vì dùng cùng data và cùng compute, nhưng không có latent bottleneck.

### Evaluation có công bằng không?

Có, vì:

- Cùng base model.
- Cùng train/test split.
- Cùng epoch budget.
- Cùng prompt policy.
- Cùng answer matcher.
- Direct SFT và LIVR restore từ cùng initial state.

### Có leakage không?

Không có leakage từ `query`, vì `query` là input chính thức và không chứa answer. Tuy nhiên, đây là domain-adapted mini evaluation trên public labels, không phải official zero-shot leaderboard.

### Vì sao LIVR thắng CV-Bench nhưng thua MathVista?

CV-Bench gần với spatial/perception reasoning, hợp với latent visual bottleneck. MathVista cần toán học chi tiết và option-level discrimination; bottleneck 2 epoch có thể chưa đủ giữ chi tiết cần thiết, nên Direct SFT mạnh hơn ở multiple-choice.

### Stage 1 sanity có ý nghĩa gì?

Nó kiểm tra xem khi answer không được nhìn trực tiếp image tokens, latent tokens có mang đủ thông tin để trả lời không. Stage 1 accuracy cao hơn random cho thấy latent tokens có học, nhưng drop lớn cho thấy chưa đủ thay thế ảnh gốc.

### Kết quả có claim được LIVR tốt hơn SFT không?

Nên nói cân bằng:

- Có trên CV-Bench.
- Không trên MathVista.
- Trung bình hai dataset có cải thiện nhẹ.
- Kết luận mạnh nhất là LIVR có tín hiệu tích cực nhưng phụ thuộc domain.

## 15. Tóm Tắt Một Phút Để Nói Với Thầy

> Em xây dựng một controlled mini-evaluation cho LIVR trên Kaggle T4. Em so sánh ba mốc: zero-shot, Direct SFT, và LIVR. Direct SFT và LIVR dùng cùng Qwen2.5-VL-3B, cùng 1000 mẫu train, cùng 5 epoch, và cùng split stratified. LIVR dùng 16 latent tokens với 2 epoch bottleneck và 3 epoch standard mask. Kết quả cho thấy LIVR vượt Direct SFT trên CV-Bench 5.5 điểm, nhưng thua trên MathVista 3 điểm. Phân tích chi tiết cho thấy LIVR tốt hơn ở free-form MathVista nhưng kém hơn ở multiple-choice. Stage 1 sanity giảm mạnh, đặc biệt MathVista, nên latent tokens có học thông tin nhưng chưa đủ giữ toàn bộ chi tiết toán học. Vì vậy em kết luận LIVR có tín hiệu tích cực cho spatial/perception reasoning, nhưng cần thêm tuning để ổn định trên visual math.

## 16. Đào Sâu Kỹ Thuật Code Và Runtime Pipeline

Phần này giải thích sâu hơn ở mức code/runtime, để khi thầy hỏi không chỉ trả lời được "em chạy notebook", mà còn giải thích được vì sao từng khối code tồn tại và nó xử lý vấn đề gì.

### 16.1. Luồng Chạy Tổng Thể Của Notebook

Notebook được tổ chức theo pipeline có trạng thái rõ ràng:

1. Cài dependencies và pin version.
2. Load dataset.
3. Load base model Qwen2.5-VL-3B ở 4-bit.
4. Gắn LoRA và latent tokens.
5. Snapshot trạng thái ban đầu.
6. Format train/test data.
7. Chạy zero-shot.
8. Restore snapshot và train Direct SFT.
9. Restore snapshot và train LIVR.
10. Patch inference Stage 1 sanity.
11. Evaluate LIVR Stage 2 và Stage 1 sanity.
12. Tổng hợp 3-way comparison.

Điểm kỹ thuật quan trọng là notebook dùng **một model object duy nhất** để tiết kiệm VRAM. Vì vậy mọi branch thí nghiệm phải restore lại từ snapshot ban đầu. Nếu không restore, Direct SFT có thể làm thay đổi LoRA rồi LIVR train tiếp trên trạng thái đã fine-tune, khiến so sánh không còn công bằng.

### 16.2. Vì Sao Không Load Ba Model Riêng

Một cách đơn giản về mặt logic là load ba model:

- Base zero-shot.
- Direct SFT model.
- LIVR model.

Nhưng trên Kaggle T4, Qwen2.5-VL-3B dù 4-bit vẫn rất nặng. Load nhiều bản model cùng lúc dễ OOM. Notebook chọn cách:

```python
initial_trainable_state = {
    n: p.detach().clone().cpu()
    for n, p in model.named_parameters()
    if p.requires_grad and is_lora_param(n)
}
initial_latent_embeddings = model.get_input_embeddings().weight[manager.latent_token_ids].detach().clone().cpu()
```

Sau đó trước mỗi branch:

```python
restore_initial_trainable_state(model, initial_trainable_state)
```

Cách này có hai lợi ích:

- Tiết kiệm VRAM vì chỉ có một model trong GPU.
- Vẫn giữ fairness vì LoRA và latent rows được trả về trạng thái ban đầu.

### 16.3. Vì Sao Không Zero LoRA Thủ Công

Một lỗi ban đầu là zero LoRA để "khôi phục base". Đây là cách nguy hiểm.

LoRA layer thường có dạng:

```text
W_eff = W_base + B @ A * scale
```

Nếu zero cả `A` và `B`, gradient có thể bị nghẽn hoặc rất yếu tùy công thức và implementation, vì hai nhánh nhân nhau. Cách đúng hơn là snapshot lại trạng thái LoRA sau khi PEFT khởi tạo, rồi restore đúng tensor ban đầu.

Vì vậy notebook dùng snapshot, không dùng:

```python
param.data.zero_()
```

Đây là một điểm thầy có thể hỏi vì nó liên quan trực tiếp đến vì sao loss LIVR trước đó không giảm.

### 16.4. Data Formatting: Từ Raw Dataset Sang Qwen Chat Conversation

Mỗi item dataset được chuyển thành dạng conversation cho Qwen2.5-VL:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
    },
]
```

Khi train, conversation gồm cả assistant answer để tạo labels.

Khi eval/generate, `include_labels=False`, nên conversation chỉ giữ user message. Điều này đảm bảo model không thấy đáp án trong lúc inference.

### 16.5. Cách Tạo Labels Cho Supervised Fine-Tuning

Trong `prepare_vqa_inputs`, notebook tạo hai text:

- `full_text`: gồm user + assistant answer, dùng để tính full input ids.
- `prompt_text`: chỉ gồm user + generation prompt, dùng để biết prompt dài đến đâu.

Sau đó labels được tạo bằng cách clone input ids và mask phần prompt:

```python
labels = full_inputs["input_ids"].clone()
labels[:, :prompt_len] = -100
```

Ý nghĩa:

- Token của prompt không bị tính loss.
- Loss chỉ tính trên answer tokens.
- Đây là setup chuẩn của causal LM SFT.

Nếu không mask prompt, model sẽ bị ép học lại cả câu hỏi/prompt, gây lệch loss và tốn capacity.

### 16.6. Chèn Latent Tokens Ở Đâu

Với LIVR, latent tokens được nối vào cuối text của user message:

```python
content_item["text"] = f"{content_item['text'].strip()}\n{latent_str}"
```

Direct SFT gọi:

```python
latent_tokens=None
```

LIVR gọi:

```python
latent_tokens=manager.latent_tokens
```

Nhờ vậy Direct SFT và LIVR dùng cùng helper, chỉ khác việc có hay không có latent tokens.

### 16.7. Stage 1 Bottleneck Mask Hoạt Động Như Thế Nào

Trong Stage 1, mục tiêu là ép thông tin ảnh đi qua latent tokens. Code xác định các vùng token:

```text
[image tokens] [prompt text tokens] [latent tokens] [answer tokens]
```

Mask Stage 1 chặn:

- Answer tokens nhìn trực tiếp image tokens.
- Prompt tokens nhìn trực tiếp image tokens.

Nhưng cho phép:

- Latent tokens nhìn prompt/image theo cấu trúc mask.
- Answer tokens nhìn latent tokens.

Ý tưởng là answer không thể lấy thông tin ảnh trực tiếp. Nếu muốn trả lời đúng, model phải đẩy thông tin thị giác cần thiết vào latent tokens.

Trong code, mask là ma trận `seq_len x seq_len`, sau đó expand theo batch:

```python
attention_mask = torch.stack(custom_masks, dim=0)
```

### 16.8. Vì Sao Cần Tính Position IDs / RoPE Khi Dùng Mask 4D

Qwen2.5-VL dùng M-RoPE cho multimodal tokens. Khi dùng attention mask 2D thông thường, model tự xử lý position ids. Nhưng khi ta thay attention mask bằng mask 4D custom, generation/forward của Qwen có thể không tự suy ra position ids đúng.

Vì vậy patch `mask_kaggle.py` phải gọi:

```python
backbone.get_rope_index(...)
```

để lấy `position_ids` trước khi thay attention mask 2D bằng mask 4D.

Nếu không làm điều này, có thể gặp lỗi shape hoặc position embedding sai.

### 16.9. Vì Sao Có Lỗi `mrope_position_ids`

Một số version Qwen/Transformers trả về thêm `mrope_position_ids` hoặc `rope_deltas`, nhưng signature của forward thay đổi theo version. Nếu truyền một argument mà forward không nhận, sẽ lỗi:

```text
unexpected keyword argument 'mrope_position_ids'
```

Fix đúng là inspect signature:

```python
if forward_supports("rope_deltas"):
    kwargs["rope_deltas"] = rope_deltas
elif forward_supports("mrope_position_ids"):
    kwargs["mrope_position_ids"] = rope_deltas
```

Nếu không hỗ trợ thì không truyền.

Điểm cần nói với thầy: đây không phải lỗi thuật toán LIVR, mà là vấn đề tương thích API của Qwen2.5-VL/Transformers.

### 16.10. Vì Sao Direct SFT Không Nên Đi Qua Custom Mask

Ban đầu patch forward can thiệp mọi forward có labels. Điều này làm cả Direct SFT cũng đi qua logic tính mask/position ids, dù Direct SFT không cần bottleneck.

Sau đó sửa thành:

```python
if stage == 1 and labels is not None and input_ids is not None:
    ... custom bottleneck mask ...
```

Như vậy:

- Direct SFT luôn đi qua forward gốc.
- LIVR Stage 2 cũng đi qua forward gốc.
- Chỉ LIVR Stage 1 mới dùng bottleneck mask.

Đây là thay đổi rất quan trọng để baseline Direct SFT sạch và ít rủi ro runtime.

### 16.11. Vì Sao Stage 1 Sanity Cần Patch Riêng Khi Generate

Patch training Stage 1 kích hoạt khi có `labels`. Nhưng trong inference:

```python
model.generate(...)
```

không có labels, nên patch training không chạy. Nếu muốn đánh giá Stage 1 sanity, phải patch forward khi generate để áp bottleneck mask trong prefill/generation.

Do đó có Cell 6b:

```python
patch_model_for_livr_inference(...)
```

Cell này không dùng cho kết quả chính LIVR Stage 2, mà chỉ dùng cho sanity check.

### 16.12. Vì Sao Gặp Lỗi `last dimension must be contiguous`

PyTorch SDPA yêu cầu attention mask/bias có memory layout contiguous ở last dimension. Custom mask tạo từ stack/to đôi khi không đáp ứng stride yêu cầu.

Lỗi:

```text
RuntimeError: (*bias): last dimension must be contiguous
```

Fix runtime:

```python
kwargs["attention_mask"] = (...).contiguous()
```

Nếu patch cũ còn nằm trong RAM, cần unwrap hoặc hotfix SDPA:

```python
F.scaled_dot_product_attention = livr_contiguous_sdpa
```

Điểm cần giải thích: đây là vấn đề tensor memory layout của PyTorch SDPA, không phải kết quả model sai.

### 16.13. Vì Sao Có NaN/Inf Loss Ở Một Vài Batch

Trong FP16 training, một số sample có thể gây overflow hoặc unstable loss, đặc biệt VLM với ảnh, long sequence, hoặc sample khó. Notebook xử lý:

```python
if torch.isnan(loss.detach()) or torch.isinf(loss.detach()):
    optimizer.zero_grad(set_to_none=True)
    skip batch
```

Điều này tốt hơn là để toàn bộ training crash.

Trong run, NaN/Inf chỉ xuất hiện ở vài step cố định. Loss history vẫn giảm đều:

```text
Direct SFT: 0.2748 -> 0.0319
LIVR:       0.4433 -> 0.0695
```

Vì vậy có thể nói đây là numerical instability cục bộ, không phải training failure.

### 16.14. Vì Sao `Scale` Của GradScaler Giảm

`Scale` là hệ số loss scaling của AMP. Khi dùng FP16, gradient nhỏ có thể underflow về 0, nên PyTorch nhân loss lên trước backward.

Nếu phát hiện overflow/Inf, GradScaler giảm scale. Ví dụ:

```text
Scale 4096 -> 2048 -> 1024
```

Scale giảm không phải lỗi. Nó là cơ chế tự bảo vệ. Miễn là:

- Training tiếp tục.
- Loss giảm.
- Không có NaN/Inf liên tục.

thì run vẫn ổn.

### 16.15. Vì Sao Trainable Params Phải Ở Float32

Khi dùng QLoRA 4-bit, base model lượng tử hóa, nhưng LoRA adapters và latent embeddings là phần trainable. Nếu phần trainable này ở FP16, `GradScaler.unscale_` sẽ báo:

```text
Attempting to unscale FP16 gradients
```

Do đó cần:

```python
if param.requires_grad and param.dtype != torch.float32:
    param.data = param.data.float()
```

Đây là trick runtime quan trọng để QLoRA + AMP chạy ổn trên Kaggle.

### 16. Vì Sao Prediction Logs Quan Trọng

Notebook lưu từng prediction vào `eval_details_*.json`:

```json
{
  "question": "...",
  "ground_truth": "...",
  "original_ground_truth": "...",
  "answer_aliases": [...],
  "model_prediction": "...",
  "is_correct": true
}
```

Nhờ log này ta có thể kiểm tra:

- Model có trả lời rỗng không.
- Model có trả lời quá dài không.
- Matcher có chấm sai không.
- LIVR sai ở loại câu nào.
- Direct SFT và LIVR khác nhau ở những câu nào.

Chính từ log này ta phát hiện MathVista LIVR không yếu toàn diện, mà thua chủ yếu ở multiple-choice.

## 17. Phân Tích Sâu Kết Quả MathVista Từ Logs

MathVista tổng:

```text
Direct SFT: 136/200 = 68.0%
LIVR Stage 2: 130/200 = 65.0%
```

So từng câu:

```text
SFT đúng, LIVR sai: 13 câu
SFT sai, LIVR đúng: 7 câu
Net: LIVR kém 6 câu
```

Theo loại câu:

```text
Free-form n=92
  Zero-shot: 30
  Direct SFT: 52
  LIVR Stage 2: 54
  LIVR hơn SFT: +2 câu

Multiple-choice n=108
  Zero-shot: 80
  Direct SFT: 84
  LIVR Stage 2: 76
  LIVR kém SFT: -8 câu
```

Vậy kết luận chính xác hơn là:

> LIVR không thua Direct SFT trên toàn bộ MathVista. LIVR thậm chí tốt hơn ở free-form, nhưng mất nhiều câu ở multiple-choice, làm tổng accuracy thấp hơn.

Điều này rất quan trọng khi thầy hỏi "vì sao LIVR MathVista thấp". Câu trả lời không nên là "LIVR kém MathVista", mà nên là:

> LIVR bị giảm ở nhóm multiple-choice/option-level discrimination, còn nhóm free-form thì không giảm.

## 18. Bộ Câu Hỏi Dự Phòng Kiểu Seminar Và Cách Trả Lời

Phần này giả lập vai thầy giáo đang hỏi sâu khi seminar đồ án.

### Câu 1. Em nói evaluation công bằng. Công bằng ở đâu?

**Trả lời:**

Em đảm bảo công bằng ở 5 điểm: cùng base model Qwen2.5-VL-3B, cùng train/test split, cùng số epoch tổng là 5, cùng prompt policy/evaluation matcher, và Direct SFT với LIVR đều restore từ cùng initial LoRA snapshot. Vì vậy khác biệt chính giữa hai nhánh là có hay không có latent tokens và bottleneck Stage 1.

### Câu 2. Vì sao không so LIVR trực tiếp với zero-shot là đủ?

**Trả lời:**

So với zero-shot không đủ vì improvement có thể chỉ đến từ fine-tuning. Direct SFT là baseline cần thiết để tách riêng lợi ích của cơ chế LIVR khỏi lợi ích chung của supervised fine-tuning.

### Câu 3. Em dùng MathVista `query`, vậy có leakage không?

**Trả lời:**

Không. `query` là input chính thức của MathVista, gồm câu hỏi, hint format và choices. Nó không chứa đáp án đúng. Leakage chỉ xảy ra nếu đưa trường `answer` hoặc thông tin hậu nghiệm của test vào prompt. Ở eval, `include_labels=False`, nên assistant answer bị bỏ khỏi input.

### Câu 4. Vì sao dùng stratified split thay vì lấy mẫu đầu/cuối?

**Trả lời:**

Vì CV-Bench được sắp theo cụm source/task. Nếu lấy 500 đầu và 200 cuối, train chủ yếu là 2D ADE20K còn test chủ yếu là 3D Omni3D Distance, gây domain shift nhân tạo. Stratified split theo `type/source/task` giúp train/test đại diện hơn cho cùng phân phối mini-benchmark.

### Câu 5. Kết quả này có phải official benchmark score không?

**Trả lời:**

Không. Đây là domain-adapted mini evaluation trên Kaggle, dùng 500 train và 200 test mỗi dataset. Nó không phải official leaderboard protocol của CV-Bench hay MathVista, và không nên so trực tiếp với kết quả leaderboard.

### Câu 6. Vì sao LIVR thắng CV-Bench?

**Trả lời:**

CV-Bench thiên về spatial/depth/object reasoning, gần với loại perception-heavy tasks mà LIVR hướng đến. Bottleneck Stage 1 ép latent tokens học thông tin thị giác hữu ích, nên khi Stage 2 mở lại full vision, model có thêm biểu diễn latent hỗ trợ trả lời. Kết quả là LIVR đạt 86.0%, hơn Direct SFT 80.5%.

### Câu 7. Vì sao LIVR thua MathVista?

**Trả lời:**

MathVista yêu cầu toán chi tiết, đọc biểu đồ, hình học và phân biệt option. Phân tích log cho thấy LIVR không thua ở free-form; nó đạt 54/92, cao hơn Direct SFT 52/92. Nhưng LIVR thua ở multiple-choice: 76/108 so với Direct SFT 84/108. Vậy vấn đề chính là option-level discrimination trong MathVista, không phải toàn bộ visual math.

### Câu 8. Stage 1 sanity giảm mạnh thì có nghĩa LIVR thất bại không?

**Trả lời:**

Không hẳn. Stage 1 sanity là stress test rất khắt khe: answer không được nhìn trực tiếp image tokens. Drop lớn cho thấy latent tokens chưa thay thế hoàn toàn ảnh gốc, đặc biệt trên MathVista. Nhưng CV-Bench Stage 1 vẫn đạt 68.5%, tức latent tokens có học một phần thông tin thị giác. Kết luận đúng là latent bottleneck có tín hiệu nhưng chưa đủ mạnh trong mini setting.

### Câu 9. Nếu Stage 1 sanity của CV-Bench bằng zero-shot 68.5%, em diễn giải thế nào?

**Trả lời:**

Em sẽ không claim quá mức. Stage 1 sanity bằng zero-shot cho thấy khi bị bottleneck, LIVR vẫn giữ được mức performance tương đương base model trên CV-Bench, nhưng chưa chứng minh latent tokens một mình vượt base. Tín hiệu mạnh hơn nằm ở Stage 2: LIVR đạt 86.0%, cao hơn Direct SFT 80.5%. Stage 1 sanity chỉ là kiểm tra phụ để xem bottleneck có làm model sập hoàn toàn không.

### Câu 10. Vì sao Direct SFT loss thấp hơn LIVR loss?

**Trả lời:**

Direct SFT được nhìn ảnh trực tiếp trong toàn bộ 5 epoch, nên objective dễ hơn. LIVR có 2 epoch đầu bị bottleneck, answer không được truy cập ảnh trực tiếp, nên loss cao hơn là hợp lý. Quan trọng là cả hai loss đều giảm đều, chứng tỏ training hoạt động.

### Câu 11. NaN/Inf batch có làm kết quả không đáng tin không?

**Trả lời:**

Không làm vô nghĩa kết quả, vì số batch NaN/Inf rất ít so với tổng 5000 step mỗi nhánh. Training loop skip batch lỗi, reset grad, và loss vẫn giảm ổn định. Tuy nhiên đây là hạn chế kỹ thuật do FP16/QLoRA trên T4, cần ghi nhận trong báo cáo.

### Câu 12. Vì sao phải pin `transformers==4.51.3`?

**Trả lời:**

Vì Kaggle có thể cài `transformers 5.0.0`, trong đó API của Qwen2.5-VL generation/forward thay đổi. Điều này gây lỗi không tương thích như argument `mrope_position_ids` hoặc prepare generation. Pin version giúp pipeline reproducible.

### Câu 13. Vì sao phải ép trainable params về float32?

**Trả lời:**

Trong QLoRA, base model lượng tử hóa 4-bit nhưng LoRA/latent embeddings là trainable. Nếu trainable gradients ở FP16, `GradScaler.unscale_` sẽ lỗi. Ép trainable params về FP32 giúp AMP + GradScaler hoạt động ổn định, trong khi base model vẫn tiết kiệm VRAM nhờ 4-bit.

### Câu 14. Em có chắc task prefix không làm bias kết quả không?

**Trả lời:**

Task prefix được áp dụng đồng đều cho zero-shot, Direct SFT và LIVR. Nó chỉ chuẩn hóa format trả lời, không cung cấp đáp án. Vì cùng prompt policy được dùng cho mọi method, nó không thiên vị riêng LIVR.

### Câu 15. Vì sao MathVista zero-shot đã 55%, khá cao?

**Trả lời:**

Do task prefix và `query` chính thức giúp model trả lời ngắn, giảm lỗi format. Ngoài ra test split có 108 multiple-choice, trong đó Qwen2.5-VL vốn đã khá mạnh. Vì thế zero-shot cao là hợp lý, nhưng Direct SFT vẫn tăng lên 68%, chứng tỏ fine-tuning có tác dụng.

### Câu 16. Vì sao CV-Bench zero-shot là 68.5%, Stage 1 sanity cũng 68.5%, nhưng LIVR Stage 2 là 86%?

**Trả lời:**

Stage 2 là chế độ inference chính của LIVR: model được nhìn ảnh gốc và latent tokens. Stage 1 sanity là chế độ bị bịt đường nhìn trực tiếp ảnh, nên khó hơn nhiều. Việc Stage 2 cao hơn rõ cho thấy biểu diễn latent có ích khi kết hợp với ảnh gốc, còn Stage 1 sanity chỉ kiểm tra khả năng latent-only pathway.

### Câu 17. Tại sao không train riêng từng dataset?

**Trả lời:**

Trong run này em chọn combined train để mô phỏng multi-task/domain-adapted setting và tiết kiệm compute. Tuy nhiên đây cũng là hạn chế. Hướng tiếp theo là train riêng CV-Bench và MathVista để xem LIVR phù hợp domain nào hơn.

### Câu 18. Nếu thầy nói LIVR chỉ hơn trung bình 1.25%, có đáng kể không?

**Trả lời:**

Em sẽ nói đây là tín hiệu tích cực nhưng chưa đủ để claim mạnh. Kết quả đáng chú ý nhất là improvement rõ trên CV-Bench +5.5%. Trung bình +1.25% cho thấy lợi ích chưa nhất quán. Vì vậy kết luận của em là LIVR có tiềm năng theo domain, không phải universal improvement trong mini setting.

### Câu 19. Nếu thầy hỏi vì sao không dùng statistical significance?

**Trả lời:**

Do giới hạn compute, run hiện tại chỉ một seed và 200 test samples mỗi dataset. Em không claim significance thống kê mạnh. Để kiểm chứng chắc hơn cần chạy nhiều seed hoặc bootstrap confidence interval trên eval logs.

### Câu 20. Nếu thầy hỏi điểm yếu lớn nhất của evaluation là gì?

**Trả lời:**

Điểm yếu lớn nhất là quy mô nhỏ và chỉ một seed. Ngoài ra MathVista `testmini` được dùng như domain adaptation mini, không phải official leaderboard protocol. Tuy nhiên evaluation vẫn có giá trị vì nó là controlled comparison giữa Direct SFT và LIVR dưới cùng điều kiện.

### Câu 21. Nếu thầy hỏi em học được gì từ lỗi runtime?

**Trả lời:**

Em học được rằng với VLM lớn, phần khó không chỉ là thuật toán mà còn là tương thích runtime: QLoRA precision, GradScaler, Qwen M-RoPE, generation cache, SDPA mask layout. Em đã phải kiểm soát dtype, position ids, attention mask contiguity, và version pinning để pipeline chạy ổn định trên T4.

### Câu 22. Nếu thầy hỏi tại sao Stage 1 inference phải monkey patch?

**Trả lời:**

Vì `model.generate()` không đi qua cùng đường training có labels. Patch training chỉ kích hoạt trong forward có labels. Để kiểm tra Stage 1 sanity khi generate, em phải patch forward inference để đưa bottleneck mask vào prefill/generation.

### Câu 23. Nếu thầy hỏi có rủi ro monkey patch làm sai kết quả không?

**Trả lời:**

Có rủi ro, nên em chỉ dùng monkey patch inference cho Stage 1 sanity, không dùng cho kết quả chính Stage 2. Kết quả chính LIVR Stage 2 đi qua forward/generate bình thường. Stage 1 sanity được xem là phân tích phụ, không phải metric chính.

### Câu 24. Nếu thầy hỏi vì sao không dùng full fine-tuning?

**Trả lời:**

Full fine-tuning Qwen2.5-VL-3B trên Kaggle T4 gần như không khả thi do VRAM. QLoRA là lựa chọn hợp lý để fine-tune parameter-efficient trong giới hạn 15GB VRAM.

### Câu 25. Nếu thầy hỏi kết luận cuối cùng của em là gì?

**Trả lời:**

Kết luận của em là: trong mini evaluation có kiểm soát, LIVR cho tín hiệu tích cực rõ trên CV-Bench, nhưng chưa ổn định trên MathVista. Điều này ủng hộ ý tưởng latent bottleneck cho spatial/perception reasoning, đồng thời chỉ ra rằng visual math cần thêm tuning hoặc schedule khác để latent tokens giữ được thông tin chi tiết hơn.

## 19. Checklist Phòng Thủ Khi Thuyết Trình

Khi thuyết trình, nên nhớ các điểm sau:

- Không nói "em reproduce paper hoàn toàn".
- Nói "mini controlled evaluation inspired by LIVR".
- Luôn nhấn mạnh Direct SFT là baseline chính.
- Nói rõ LIVR thắng CV-Bench nhưng thua MathVista.
- Không né MathVista; giải thích bằng phân tích free-form vs multiple-choice.
- Nói Stage 1 sanity là stress test phụ, không phải kết quả chính.
- Nếu bị hỏi leakage, nhấn mạnh `query` là input chính thức, không chứa answer.
- Nếu bị hỏi fairness, nói restore snapshot, same data, same epochs, same prompt, same matcher.
- Nếu bị hỏi runtime errors, giải thích QLoRA/AMP/M-RoPE/SDPA là các vấn đề triển khai trên T4.
- Nếu bị hỏi claim, nói claim vừa phải: LIVR có tín hiệu tích cực theo domain, chưa phải universal improvement.
