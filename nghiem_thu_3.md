# BÁO CÁO NGHIỆM THU GIAI ĐOẠN 3: TRIỂN KHAI MATHVISTA VỚI CHIA TÁCH ĐỘNG, KHẮC PHỤC LỖI THIẾU ẢNH VÀ CHUẨN HÓA PIPELINE ĐÁNH GIÁ KHOA HỌC

Tài liệu này tổng hợp toàn bộ các vấn đề phát sinh trong quá trình đánh giá hiệu năng (Evaluation) mô hình **LIVR-Mini-Benchmark** trên hai tập dữ liệu benchmark độc lập là **MathVista** và **CV-Bench** (chạy trên môi trường Kaggle/Colab), cùng các giải pháp kỹ thuật triệt để đã được đóng gói và cập nhật thành công lên nhánh `develop`.

---

## 1. Tổng hợp các Vấn đề và Giải pháp đã triển khai

Các thay đổi trong giai đoạn này tập trung vào việc ổn định luồng dữ liệu (Data Pipeline), giải quyết lỗi thiếu dữ liệu hình ảnh, ngăn chặn rò rỉ thông tin dữ liệu kiểm thử (Data Leakage) và nâng cao độ chính xác khi đối khớp câu trả lời trắc nghiệm (MCQ).

| STT | Vấn đề phát hiện | Nguyên nhân | Giải pháp kỹ thuật | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Lỗi thụt lề (Indentation Error) tại Cell 6 của các notebook | Biến cấu hình đường dẫn `output_dir` bị thụt lề sai cú pháp thụt đầu dòng (indentation) của Python khiến notebook báo lỗi biên dịch ngay khi chạy. | Chuẩn hóa thụt lề, căn chỉnh thẳng hàng sát lề trái cho các biến toàn cục trong cả hai notebook đánh giá. | **Đã sửa đổi** |
| **2** | Thiếu file nhật ký đánh giá chi tiết (Log Details JSON) | Tiến trình huấn luyện tự động `train_kaggle.py` chỉ hiển thị điểm tổng hợp mà không ghi nhận chi tiết dự đoán từng mẫu của mô hình baseline và LIVR. | Tích hợp cơ chế tự động xuất log chi tiết dạng JSON (`train_eval_details_livr.json` và `train_eval_details_base.json`) ghi nhận cụ thể từng câu hỏi, nhãn gốc, và dự đoán của mô hình. | **Đã sửa đổi** |
| **3** | Sự mâu thuẫn tri thức đa phương thức của tập Train `VisuLogic` | Tập `VisuLogic-Train` gốc trên Hugging Face chỉ chứa dữ liệu văn bản thuần (Text-only), không có ảnh đi kèm. Việc này đi ngược lại với triết lý huấn luyện biểu diễn hình ảnh (Image-guided bottleneck) của LIVR. | Thay thế benchmark VisuLogic bằng **MathVista** (sử dụng split `testmini` gồm 1.000 mẫu có ảnh đa phương thức 100%), cấu hình đường dẫn dataset mới tại [config/evaluation_config.json](file:///home/ductien/Documents/LIVR-Mini-Benchmark/config/evaluation_config.json). | **Đã sửa đổi** |
| **4** | Lỗi nạp dữ liệu PIL và crash khi gọi processor với mẫu không có ảnh | Lớp VQA `prepare_vqa_inputs` mặc định mong muốn nhận mảng ảnh hợp lệ, khi gặp ảnh rỗng hoặc danh sách rỗng (`[[]]`) sẽ báo lỗi `TypeError`. | Định nghĩa cục bộ `prepare_vqa_inputs` trong notebook: Tự động lọc các ảnh bị trống (`None`), chỉ thêm thẻ `image` khi có ảnh thực tế và truyền `images=None` cho processor nếu đó là mẫu tin chỉ có chữ. | **Đã sửa đổi** |
| **5** | Rò rỉ dữ liệu kiểm thử (Data Leakage) trên các tập đơn split | Cả `CV-Bench` và `MathVista (testmini)` đều là các tập dữ liệu đơn split (không có sẵn split train/test riêng biệt). Nếu lấy trùng mẫu cho bước thích ứng và đánh giá sẽ làm sai lệch kết quả. | Thiết lập cơ chế chia tách chỉ mục động trong hàm `prepare_eval_dataset` của Cell 6: Trích xuất **300 mẫu đầu tiên** làm tập huấn luyện thích ứng và **50 mẫu cuối cùng** làm tập kiểm định độc lập. | **Đã sửa đổi** |
| **6** | Đánh giá sai lệch điểm trắc nghiệm (MCQ Evaluation Discrepancies) | Mô hình trả lời đúng nhưng theo các định dạng lệch chuẩn như `(A)`, `a.`, hoặc viết nguyên câu dài chứa chữ cái lựa chọn, dẫn tới bị chấm điểm sai khi so khớp bằng phép `==` thông thường. | Xây dựng hàm so khớp thông minh `match_answer(pred, target)` sử dụng **Regular Expression (Regex)** để chuẩn hóa văn bản và kiểm tra sự xuất hiện độc lập của chữ cái đáp án đúng. | **Đã sửa đổi** |

---

## 2. Chi tiết Kỹ thuật áp dụng & Ý nghĩa Khoa học

### A. Tích hợp MathVista và cấu hình `testmini`
MathVista (`AI4Math/MathVista`) là một benchmark chuẩn mực đo lường khả năng lập luận toán học thị giác cực kỳ nổi tiếng. Chúng tôi sử dụng split `testmini` (1.000 mẫu) của MathVista, phân chia động:
*   **Tập thích ứng (Domain Adaptation)**: Lấy 300 mẫu đầu tiên để tinh chỉnh thích ứng lora (Stage 2).
*   **Tập đánh giá độc lập (Validation)**: Lấy 50 mẫu cuối cùng để kiểm thử accuracy.

```python
# Nạp dữ liệu trong Cell 4
mathvista_dataset = load_dataset("AI4Math/MathVista", split="testmini", cache_dir=cache_dir)
```

*   **Ý nghĩa kỹ thuật & khoa học**: Đảm bảo toàn bộ quá trình thích ứng miền (Stage 2) diễn ra trên dữ liệu có ảnh gốc đầy đủ (Multimodal). Điều này giúp LoRA adapters học cách phối hợp tốt nhất các vector đặc trưng thị giác từ ảnh thực tế và các Latent Tokens đã được huấn luyện ở Giai đoạn 1.

---

### B. Hàm `prepare_eval_dataset` hỗ trợ linh hoạt PIL Image và Decoded Image
Trong MathVista, ảnh được trả về từ thư viện `datasets` dưới dạng đối tượng PIL Image hoặc thông tin đường dẫn. Hàm nạp dữ liệu động được tối ưu để thích nghi với mọi cấu trúc:

```python
# Trích xuất ảnh (MathVista dùng decoded_image hoặc image)
image_obj = item.get('decoded_image') or item.get('image')
if isinstance(image_obj, str) and image_obj:
    img_path = os.path.join(cache_dir, image_obj)
    if os.path.exists(img_path):
        from PIL import Image
        image_obj = Image.open(img_path).convert("RGB")
elif image_obj is not None:
    from PIL import Image
    if isinstance(image_obj, Image.Image):
        image_obj = image_obj.convert("RGB")
```

*   **Ý nghĩa kỹ thuật**: Giúp tương thích hoàn hảo với cấu trúc lưu trữ ảnh đa dạng của các benchmark khác nhau. Hàm tự động chuẩn hóa ảnh về hệ màu `RGB` và bọc trong đối tượng PIL Image tiêu chuẩn trước khi chuyển tới processor của mô hình.

---

### C. Ngăn chặn Data Leakage bằng phân tách Split động
Trong hàm `prepare_eval_dataset` (Cell 6), chúng tôi đưa vào logic phân chia chỉ mục dựa trên loại split thực tế:

```python
# Chia tách động để tránh rò rỉ dữ liệu (data leakage) trên các tập đơn split
if is_single_split and is_train:
    split_data = dataset_split.select(range(min(num_samples, len(dataset_split))))
elif is_single_split and not is_train:
    start_idx = max(0, len(dataset_split) - num_samples)
    split_data = dataset_split.select(range(start_idx, len(dataset_split)))
```

*   **Ý nghĩa khoa học**: Việc chia nhỏ một tập duy nhất thành tập con để học thích ứng (Adaptation) và tập con để chấm điểm (Validation) tách biệt đảm bảo tính trung thực và khách quan của kết quả nghiên cứu, tránh hiện tượng mô hình "học vẹt" đáp án.

---

### D. Bộ đối khớp đáp án MCQ thông minh bằng Regex
Hàm `match_answer` tại **Cell 7** giúp đánh giá điểm số chuẩn xác:

```python
def match_answer(pred, target):
    pred = pred.strip().lower()
    target = target.strip().lower()
    
    # So khớp trực tiếp
    if pred == target:
        return True
        
    # Loại bỏ dấu ngoặc đơn và dấu chấm
    pred_cleaned = pred.replace('(', '').replace(')', '').replace('.', '').strip()
    target_cleaned = target.replace('(', '').replace(')', '').replace('.', '').strip()
    if pred_cleaned == target_cleaned:
        return True
        
    # Sử dụng biểu thức chính quy (Regex) để trích xuất ký tự phương án đơn lẻ
    if len(target_cleaned) == 1 and target_cleaned.isalpha():
        pattern = rf"\b{target_cleaned}\b"
        if re.search(pattern, pred_cleaned):
            return True
            
    return False
```

*   **Ý nghĩa khoa học**: Loại bỏ sự phụ thuộc vào các từ đệm sinh tự do của VLM, đánh giá đúng năng lực thực tế của mô hình và đưa ra số liệu thống kê accuracy khách quan.

---

## 3. Quy trình nghiệm thu trên môi trường Kaggle / Colab

Bạn chỉ cần thực hiện 3 bước đơn giản sau:

1.  **Đẩy code từ máy Local lên Git**:
    ```bash
    git add 02_evaluation_mini.ipynb 02_evaluation_mini_kaggle.ipynb config/evaluation_config.json nghiem_thu_3.md
    git commit -m "feat: switch evaluation benchmark to MathVista and update reports"
    git push origin develop
    ```
2.  **Kéo code trên môi trường Kaggle / Colab (Cell 1)**:
    ```bash
    !git reset --hard
    !git pull origin develop
    ```
3.  **Chạy lại toàn bộ tiến trình từ Cell 3 đến Cell 7**:
    *   **Cell 3**: Sẽ đổi tự động cấu hình sang `math_vista` và tải cấu hình Kaggle.
    *   **Cell 4**: Sẽ tải tập dữ liệu `MathVista (testmini)` gồm 1.000 mẫu có ảnh đi kèm và CV-Bench.
    *   **Cell 6**: Sẽ tiến hành huấn luyện thích ứng (Stage 2) trên tập kết hợp (CV-Bench 300 mẫu đầu + MathVista 300 mẫu đầu) có hình ảnh đầy đủ.
    *   **Cell 7**: Chạy đánh giá độc lập (trên CV-Bench 50 mẫu cuối + MathVista 50 mẫu cuối), xuất tệp JSON kết quả chi tiết.
