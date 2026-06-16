# 📊 Báo Cáo Nghiệm Thu Kỹ Thuật & Tối Ưu Hóa Hệ Thống LIVR-Mini
**Mục tiêu**: Báo cáo chi tiết các thay đổi mã nguồn (Diffs), lý do kỹ thuật, ví dụ thực tế minh họa, và tổng hợp các sự cố/giải pháp khi triển khai mô hình trên **Google Colab Tesla T4 GPU (16GB VRAM)**.

---

## 📂 1. Tổng Quan Cấu Trúc File Đã Cập Nhật

Hệ thống đã được cập nhật đồng bộ trên nhánh `develop`:
*   **[src/model.py](file:///home/ductien/Documents/LIVR-Mini-Benchmark/src/model.py)**: Khởi tạo mô hình Qwen2.5-VL cấu hình QLoRA 4-bit, Vocab expansion ($K=16$) và unfreeze có chọn lọc bảng nhúng thông qua Hook.
*   **[src/mask.py](file:///home/ductien/Documents/LIVR-Mini-Benchmark/src/mask.py)**: Monkey-patch attention mask, chuyển đổi dữ liệu mask sang `float32` trước khi cast động sang kiểu dữ liệu của mô hình nhằm tối ưu cho GPU T4.
*   **[src/utils.py](file:///home/ductien/Documents/LIVR-Mini-Benchmark/src/utils.py)**: Lọc dữ liệu, visual de-duplication bằng pHash, cải tiến `prepare_vqa_inputs` nhận diện chế độ huấn luyện/suy luận để tự động chèn header trợ lý.
*   **[config/implement_config.json](file:///home/ductien/Documents/LIVR-Mini-Benchmark/config/implement_config.json)**: Siêu tham số huấn luyện 2 giai đoạn phù hợp với T4 (accumulation steps = 8).
*   **[config/evaluation_config.json](file:///home/ductien/Documents/LIVR-Mini-Benchmark/config/evaluation_config.json)**: Cấu hình đánh giá trên các tập dữ liệu Novel miền tri thức mới.
*   **[01_implement_mini.ipynb](file:///home/ductien/Documents/LIVR-Mini-Benchmark/01_implement_mini.ipynb)**: Notebook huấn luyện hoàn chỉnh, tích hợp biểu đồ loss và so sánh đối chiếu baseline accuracy bằng `model.disable_adapter()`.
*   **[02_evaluation_mini.ipynb](file:///home/ductien/Documents/LIVR-Mini-Benchmark/02_evaluation_mini.ipynb)**: Notebook tinh chỉnh thích ứng 2 epochs và chạy thử nghiệm Sanity Check (chặn ảnh đột ngột) xuất bảng tổng hợp.

---

## 🔍 2. Chi Tiết Các Thay Đổi (Diffs) & Giải Thích Thực Chiến

### Kỹ thuật 1: Lượng Hóa QLoRA 4-bit & Float16 Compute Type (`src/model.py`)

#### 📑 Mã nguồn thay đổi (Diff):
```python
# Cấu hình lượng hóa QLoRA 4-bit tối ưu hóa VRAM cho card T4 miễn phí
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Dùng float16 cho T4 thay vì bfloat16
)
```

#### 💡 Giải thích & Ví dụ thực tế:
*   **Vấn đề**: Bản gốc của mô hình `Qwen2.5-VL-3B-Instruct` ở độ chính xác 16-bit tiêu tốn khoảng **6GB VRAM** chỉ để tải lên bộ nhớ. Khi huấn luyện, các lớp kích hoạt (activations), bộ tối ưu hóa (optimizer states) và gradient sẽ đẩy tổng VRAM lên **vượt quá 20GB**, gây crash RAM ngay lập tức trên Colab T4 (giới hạn 16GB).
*   **Giải pháp**: QLoRA nén các tham số gốc từ 16-bit xuống định dạng **NF4 (NormalFloat 4-bit)**. 
*   **Ví dụ thực tế**: Tương tự như việc bạn mang một cuốn từ điển bách khoa toàn thư dày 1,000 trang nén lại thành một bản tóm tắt bỏ túi chỉ dài 100 trang. Các kiến thức cốt lõi vẫn được giữ nguyên. Khi cần tra cứu (forward pass), bạn chỉ giải nén tạm thời trang sách đó ra viết nháp bằng bút chì rồi xóa đi (dequantize sang `float16` trên Tensor Cores của T4), giúp mô hình chạy cực kỳ nhẹ nhàng trên VRAM chỉ tốn **~1.8GB** lúc load.

---

### Kỹ thuật 2: Đồng Bộ Kiểu Dữ Liệu & Cast Động Lớp Attention Mask (`src/mask.py`)

#### 📑 Mã nguồn thay đổi (Diff):
```diff
- float_mask = torch.zeros(seq_len, seq_len, dtype=torch.bfloat16, device=device)
+ float_mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
...
- attention_mask = torch.stack(custom_masks, dim=0)
+ attention_mask = torch.stack(custom_masks, dim=0).to(dtype=self.dtype)
```

#### 💡 Giải thích & Ví dụ thực tế:
*   **Vấn đề**: Card đồ họa Tesla T4 thế hệ cũ **không có nhân phần cứng hỗ trợ bfloat16** (Brain Float 16). Nếu bạn ép GPU khởi tạo ma trận mặt nạ attention bằng `bfloat16`, PyTorch sẽ phải giả lập phần mềm, làm chậm tốc độ tính toán từ 3 đến 5 lần, hoặc ném lỗi **Device Mismatch** khi cộng ma trận chú ý.
*   **Giải pháp**: Chúng ta khởi tạo ma trận mặt nạ bằng kiểu số thực `float32` tiêu chuẩn (đảm bảo độ chính xác toán học ổn định nhất), sau đó ở bước cuối cùng trước khi đưa vào transformer layer, ta cast động ma trận này sang kiểu dữ liệu hiện tại của mô hình (`self.dtype` - sẽ tự động là `float16` nếu chạy ở chế độ QLoRA 4-bit trên T4, và là `bfloat16` nếu chạy trên card L4/A100).
*   **Ví dụ thực tế**: Giống như việc bạn soạn thảo văn bản bằng định dạng Unicode tiêu chuẩn trên máy tính của bạn (soạn thảo bằng `float32`). Trước khi in ra gửi cho khách hàng dùng máy in đời cũ (GPU T4), bạn xuất nó ra định dạng PDF tương thích hoàn toàn để đảm bảo tài liệu không bị lỗi font hay chậm tiến độ in ấn.

---

### Kỹ thuật 3: Phân Tách Câu Hỏi Khi Đánh Giá & Cắt Lát Kết Quả (Notebooks)

#### 📑 Mã nguồn thay đổi (Diff):
```python
# Chỉ gửi câu hỏi của User sang sinh output tự hồi quy
conv_for_generation = [msg for msg in conv if msg["role"] == "user"]
...
# Cắt lát để chỉ giải mã phần chữ do mô hình tự sinh ra
input_len = inputs["input_ids"].shape[1]
pred_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
```

#### 💡 Giải thích & Ví dụ thực tế:
### Kỹ thuật 4: Tải song song đa luồng & Nạp cục bộ qua file JSON (`src/utils.py`)

#### 📑 Mã nguồn thay đổi (Diff):
```python
# Tải song song toàn bộ kho dữ liệu (chỉ lấy thư mục train)
snapshot_download(
    repo_id="Kkuntal990/LIVR_mixed",
    repo_type="dataset",
    local_dir=local_dir,
    token=token,
    max_workers=8,
    allow_patterns=["train/**"],
    local_dir_use_symlinks=False
)
# Nạp trực tiếp metadata.jsonl bằng bộ đọc JSON của Hugging Face
metadata_file = os.path.join(local_dir, "train/metadata.jsonl")
dataset = load_dataset("json", data_files=metadata_file)
# Liên kết đường dẫn ảnh tuyệt đối và cast cột sang Image chuẩn
dataset = dataset.map(lambda x: {"image": os.path.join(local_dir, "train", x["file_name"])})
dataset = dataset.cast_column("image", Image())
```

#### 💡 Giải thích & Ví dụ thực tế:
*   **Vấn đề**: Tải 9.000 file ảnh thô qua mạng với cơ chế mặc định của HF sẽ bị tải lười (Lazy-Loading), tức là đọc ảnh nào tải ảnh đó, dẫn đến 9.000 kết nối tuần tự cực kỳ chậm.
*   **Giải pháp**: Tải một lượt duy nhất toàn bộ repo thông qua đa luồng (`max_workers=8`), giới hạn chỉ tải thư mục `train/**` (~8.7 GB). Sau đó nạp trực tiếp file `metadata.jsonl` bằng bộ đọc JSON, ánh xạ đường dẫn tuyệt đối cho ảnh cục bộ và dùng `cast_column` định dạng sang kiểu `Image`. Điều này hoàn toàn bỏ qua lỗi nhận diện nhầm split của bộ nạp `imagefolder` khi gặp thư mục con trùng tên (như `images/counting/train/`).
*   **Ví dụ thực tế**: Thay vì mỗi lần nấu ăn bạn lại chạy ra siêu thị mua đúng 1 quả cà chua (Lazy Loading), bạn thuê một đội xe tải chở thẳng một tủ chứa thực phẩm đặt vào bếp nhà bạn (Snapshot Download). Bạn không dùng người giúp việc tự động sắp xếp (bộ nạp `imagefolder` thông minh nhưng hay nhận nhầm ngăn tủ `train` của tác vụ đếm làm cả cái bếp), mà bạn tự tay mở đúng cuốn sổ tay thực đơn (`metadata.jsonl`) rồi tự lấy nguyên liệu ở đúng địa chỉ trong kho để nấu ăn.

---

## 🚨 3. Các Sự Cố Đã Gặp & Giải Pháp Khắc Phục (Troubleshooting)

### 🚨 Sự cố 1: Lỗi thiết bị chạy (Device Mismatch)
*   **Triệu chứng**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
*   **Nguyên nhân**: Trong hàm monkey-patch của attention mask, các tensor `torch.ones` hoặc `torch.zeros` được tạo mặc định trên bộ nhớ **CPU** của hệ thống, trong khi các token đầu vào của mô hình đã được đẩy lên **GPU** (`cuda:0`).
*   **Khắc phục**: Thay đổi toàn bộ các dòng khởi tạo tensor trong [mask.py](file:///home/ductien/Documents/LIVR-Mini-Benchmark/src/mask.py), bổ sung thuộc tính `device=device` để ép PyTorch khởi tạo ma trận trực tiếp trên GPU đang chạy.

### 🚨 Sự cố 2: Trôi nghĩa từ vựng gốc (Token Drift)
*   **Triệu chứng**: Sau một vài epoch huấn luyện, mô hình bắt đầu mất khả năng nói tiếng Anh/Việt chuẩn, sinh ra các từ vô nghĩa hoặc câu cú lộn xộn.
*   **Nguyên nhân**: Bảng từ vựng mở rộng thêm 16 Latent Tokens. Khi unfreeze lớp nhúng `embed_tokens.weight` để tối ưu hóa vector nhúng cho các Latent Tokens này, thuật toán lan truyền ngược vô tình cập nhật cả vector nhúng của hàng nghìn từ vựng ngôn ngữ gốc của Qwen2.5-VL.
*   **Khắc phục**: Đăng ký một **Backward Hook** trên tầng nhúng. Hook này đóng vai trò như một mặt nạ nhị phân động, nhân gradient của bảng nhúng với $0.0$ tại mọi vị trí của từ vựng cũ và giữ nguyên $1.0$ tại vị trí của Latent Tokens.
    ```python
    def make_embedding_hook(ids):
        def hook(grad):
            mask = torch.zeros(grad.size(0), 1, dtype=grad.dtype, device=grad.device)
            mask[ids.to(grad.device)] = 1.0
            return grad * mask
        return hook
    ```

### 🚨 Sự cố 3: Tràn bộ nhớ VRAM khi huấn luyện 2 giai đoạn trên T4
*   **Triệu chứng**: `RuntimeError: CUDA out of memory. Tried to allocate...`
*   **Nguyên nhân**: Kích thước ảnh đầu vào của VLM rất lớn, cộng thêm việc lưu trữ các ma trận attention trung gian trong quá trình huấn luyện vượt quá 16GB bộ nhớ của T4.
*   **Khắc phục**: 
    1. Thiết lập `batch_size_per_device = 1` trong file config để chỉ xử lý 1 ảnh tại một thời điểm.
    2. Sử dụng `grad_accumulation_steps = 8` để mô phỏng batch size bằng 8.
    3. Sử dụng `torch.nn.utils.clip_grad_norm_` giới hạn `max_norm=1.0` để chống bùng nổ gradient trong quá trình tích lũy.
    4. Thêm chốt chặn gán nhãn `-100` cho tất cả token không thuộc phần trả lời của trợ lý để mô hình không phải tính toán loss và gradient cho phần câu hỏi, tiết kiệm đáng kể tài nguyên tính toán.

### 🚨 Sự cố 4: Tải ảnh quá chậm (Lazy-Loading Bottleneck)
*   **Triệu chứng**: Tải 9.000 ảnh thô về Colab mất hơn 2 tiếng đồng hồ và thường xuyên bị tạm dừng giữa chừng.
*   **Nguyên nhân**: Hàm `load_dataset` mặc định của Hugging Face chỉ tải thông tin văn bản, còn dữ liệu ảnh chỉ được kéo về từng ảnh một qua kết nối HTTP đơn luồng mỗi khi vòng lặp tiền xử lý Python chạm tới cột `image`.
*   **Khắc phục**: Chuyển đổi sang sử dụng `huggingface_hub.snapshot_download` để kéo toàn bộ repository về ổ đĩa cục bộ thông qua cơ chế song song đa luồng (`max_workers=8`) trước khi chạy code làm sạch dữ liệu.

### 🚨 Sự cố 5: Xung đột cấu trúc cột (Schema Mismatch / Different Features)
*   **Triệu chứng**: `[WARNING] Gặp lỗi khi chạy tải song song hoặc nạp cục bộ: Metadata files ... have different features ...`
*   **Nguyên nhân**: Bộ đọc `imagefolder` của Hugging Face khi trỏ vào thư mục gốc của snapshot đã tự động quét và ghép toàn bộ các tệp `metadata.jsonl` ở các thư mục con khác nhau (gồm cả tập `vsp_planning` có chứa thêm cột tọa độ mê cung `'map_id'`, `'map_desc'`, v.v.). Sự lệch pha cấu trúc cột này làm bộ đọc bị lỗi.
*   **Khắc phục**: Chỉ định rõ ràng thư mục con chứa tập huấn luyện chính: `data_dir=os.path.join(local_dir, "train")`, giúp cô lập file metadata cần đọc và nạp thành công 100%.
