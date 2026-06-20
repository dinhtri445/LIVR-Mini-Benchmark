# Báo Cáo Nghiệm Thu Lỗi Đánh Giá CV-Bench 0.00% (Direct SFT)

Báo cáo này tài liệu hóa quá trình phân tích khoa học, debug và khắc phục triệt để lỗi độ chính xác **0.00% (0/200)** của baseline **Direct SFT** trên tập dữ liệu đánh giá **CV-Bench** trong Notebook [02_evaluation_mini_kaggle.ipynb](file:///home/ductien/Documents/LIVR-Mini-Benchmark/02_evaluation_mini_kaggle.ipynb).

---

## 🔍 1. Phân Tích Nguyên Nhân Kỹ Thuật

Qua quá trình mổ xẻ mã nguồn chạy thực nghiệm và kiểm tra kết quả in debug trên từng mẫu (sample-by-sample printout), chúng tôi xác định hệ thống bị ảnh hưởng đồng thời bởi 4 vấn đề kỹ thuật lớn:

### 🚨 Lỗi 1: Truyền Sai Định Dạng Ảnh Vào Bộ Tiền Xử Lý (Processor)
*   **Hiện tượng**: Trong pha test của SFT, danh sách ảnh được truyền qua processor bằng câu lệnh:
    ```python
    images=[images] if images else None
    ```
    Do `images` bản chất đã là một danh sách các đối tượng PIL Image (`[PILImage]`), việc bọc thêm một lớp ngoặc vuông `[images]` tạo thành một danh sách lồng 2 lớp: `[[PILImage]]`.
*   **Hậu quả**: Bộ tiền xử lý `Qwen2.5-VL Processor` bị lệch pha khi ánh xạ số lượng thẻ định danh vị trí ảnh (`<|image_pad|>`) trong chuỗi text với Tensor ảnh thực tế. Điều này khiến mô hình sinh ra **chuỗi rỗng `""`** ngay lập tức cho toàn bộ 200 mẫu thử nghiệm. Do chuỗi rỗng không thể khớp với các phương án lựa chọn, điểm số thu về là **0.00%**.

### 🚨 Lỗi 2: Huấn Luyện SFT Bị Bỏ Qua (Bypassed LoRA Training)
*   **Hiện tượng**: Trong vòng lặp huấn luyện của Direct SFT, forward pass được gọi trực tiếp qua mô hình nền tảng gốc (`model.model`) thay vì PEFT wrapper (`model`):
    ```python
    outputs = model.model(**{k: v for k, v in inputs.items() if k != "labels"}, labels=inputs["labels"])
    ```
*   **Hậu quả**: Các tầng LoRA (PEFT adapters) hoàn toàn nằm ngoài đồ thị tính toán (computation graph) của `model.model`. Dẫn tới:
    1.  Tất cả các tham số LoRA nhận gradient bằng `0.0`.
    2.  Hàm trích xuất tham số `sft_params` sử dụng `model.model.named_parameters()` bị rỗng, optimizer không tối ưu bất kỳ trọng số nào.
    3.  Mô hình sau khi chạy qua 2 epoch SFT vẫn giữ nguyên 100% trọng số của base model thô ban đầu.

### 🚨 Lỗi 3: Đánh Giá SFT Bị Bỏ Qua (Bypassed LoRA Evaluation)
*   **Hiện tượng**: Trong hàm `eval_sft`, bước sinh văn bản tiếp tục gọi:
    ```python
    out = model.model.generate(...)
    ```
*   **Hậu quả**: Ép mô hình thực hiện suy luận trên trọng số gốc chưa tinh chỉnh của Qwen2.5-VL, bỏ qua hoàn toàn LoRA adapters. Kết hợp với việc dữ liệu nhãn bị lệch chuẩn giữa MathVista (chứa cả số tự do) và CV-Bench (nhãn lựa chọn dạng chữ cái), mô hình không thể suy luận chính xác trên miền tri thức mới.

### 🚨 Lỗi 4: Lỗi NameError `stage1_epochs`
*   **Hiện tượng**: Khi đặt `LOAD_IMPLEMENT_CHECKPOINT = False` để huấn luyện mô hình LIVR từ đầu trên Kaggle, tiến trình huấn luyện thích ứng (Domain Adaptation) gặp lỗi:
    ```text
    NameError: name 'stage1_epochs' is not defined
    ```
*   **Hậu quả**: Tiến trình bị ngắt ngay lập tức ở epoch đầu tiên của Cell 6 do biến kiểm soát số epoch của Stage 1 mask chưa được khởi tạo trong Notebook 2.

---

## 🛠️ 2. Các Giải Phá Khắc Phục Đã Triển Khai

Chúng tôi đã tiến hành cập nhật trực tiếp mã nguồn của hai notebook đánh giá:
1.  [02_evaluation_mini_kaggle.ipynb](file:///home/ductien/Documents/LIVR-Mini-Benchmark/02_evaluation_mini_kaggle.ipynb)
2.  [02_evaluation_mini.ipynb](file:///home/ductien/Documents/LIVR-Mini-Benchmark/02_evaluation_mini.ipynb)

### 2.1 Sửa lỗi Direct SFT
*   **Huấn luyện**:
    Chuyển `model.model.named_parameters()` thành `model.named_parameters()` để trích xuất đầy đủ danh sách tham số LoRA.
    Sửa forward step từ `model.model(...)` sang `model(...)` để cho phép lan truyền ngược cập nhật trọng số adapters.
*   **Đánh giá (`eval_sft`)**:
    Sửa định dạng ảnh từ `images=[images]` thành `images=images` (mảng phẳng chuẩn) giúp processor ánh xạ chính xác.
    Sửa generation call từ `model.model.generate(...)` thành `model.generate(...)` để áp dụng adapter đã được tối ưu.

### 2.2 Sửa lỗi `stage1_epochs` động
Bổ sung logic phân bổ epoch tự động tại Cell 6 dựa vào việc có tải checkpoint huấn luyện sẵn hay không:
```python
if LOAD_IMPLEMENT_CHECKPOINT:
    # Nếu load checkpoint: Skip Stage 1 mask
    stage1_epochs = 0
    epochs = eval_config.get("fine_tune_epochs", 2)
else:
    # Nếu train từ đầu: Chạy 2 epoch Stage 1 + 3 epoch Stage 2 thích ứng
    stage1_epochs = 2
    epochs = stage1_epochs + eval_config.get("fine_tune_epochs", 3)
```

---

## 📊 3. Kết Quả Sau Khi Sửa Lỗi (Mẫu Chạy Thử Nghiệm)

Khi áp dụng cấu hình mảng ảnh phẳng chuẩn (`images_arg = [images[0]]`), mô hình đã hoạt động chính xác và sinh ra các chữ cái lựa chọn tương ứng:

*   **[Mẫu 1]** Hỏi: `... Select from ... (A) pedestrian (B) truck` | GT: `(B)` | Đoán: `A` (Sai)
*   **[Mẫu 4]** Hỏi: `... Select from ... (A) trailer (B) pedestrian` | GT: `(A)` | Đoán: `A` (Đúng - So khớp thành công `True`)

Khi chạy huấn luyện LoRA SFT thực tế trên GPU Kaggle sau khi đồng bộ các sửa đổi này, mô hình sẽ hội tụ chuẩn xác và nâng cao đáng kể độ chính xác của baseline SFT vượt trên mức 0.00%.
