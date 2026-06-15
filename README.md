# 🧠 LIVR-Mini-Benchmark

**LIVR-Mini-Benchmark** là một dự án nghiên cứu, thực nghiệm và đánh giá phiên bản tinh gọn của kiến trúc **LIVR (Latent Implicit Visual Reasoning)** tích hợp trên mô hình đa phương thức nền tảng **Qwen2.5-VL-3B-Instruct**. 

Dự án được tối ưu hóa toàn diện để chạy ổn định và đạt hiệu năng cao trên hạ tầng phần cứng hạn chế như **Google Colab Tesla T4 GPU (16GB VRAM)** miễn phí.

---

## 🎯 1. Ý Tưởng Dự Án & Phương Pháp LIVR
Kiến trúc LIVR (Suy luận Thị giác Ngầm ẩn qua Trạng thái Ẩn) dựa trên bài báo khoa học cùng tên, hoạt động theo cơ chế ép mô hình đa phương thức mã hóa các đặc trưng thị giác của ảnh vào một chuỗi gồm $K$ trạng thái ẩn (Latent Tokens) đặc biệt trước khi đưa ra câu trả lời cuối cùng. 

Quá trình huấn luyện được thực hiện qua **2 Giai đoạn (Two-Stage Training)**:
1.  **Giai đoạn 1 (Bottleneck Mask)**: Sử dụng một ma trận Attention Mask tùy biến nhằm chặn hoàn toàn kết nối chú ý (attention) trực tiếp từ câu hỏi (Prompt) và câu trả lời (Answer) đến các mảnh ảnh (Image Tokens). Toàn bộ dòng thông tin thị giác bắt buộc phải đi vòng qua $K$ Latent Tokens để nén thông tin.
    $$A_{i, j} = \begin{cases} 0 & \text{nếu } i \ge j \text{ và } \text{chặn}(i, j) = \text{False} \\ -65500.0 & \text{ngược lại} \end{cases}$$
2.  **Giai đoạn 2 (Unmasked Co-training)**: Gỡ bỏ mặt nạ bottleneck, phục hồi ma trận Causal Attention tiêu chuẩn để mô hình đồng hóa biểu diễn ẩn cấu trúc cao đã học được ở Giai đoạn 1 với các đặc trưng ảnh chi tiết mức thấp (raw pixels).

---

## ⚡ 2. Các Tối Ưu Hóa Kỹ Thuật Cho GPU T4 (16GB VRAM)
Để có thể chạy huấn luyện mô hình Vision-Language 3 tỷ tham số trên card T4 miễn phí mà không bị lỗi tràn bộ nhớ (Out-Of-Memory - OOM) hay đạo hàm NaN:
*   **Lượng hóa QLoRA 4-bit (NF4)**: Giảm dung lượng tải trọng số mô hình nền từ ~6GB xuống chỉ còn **1.8GB VRAM**.
*   **Kiểu dữ liệu tính toán Float16**: Cấu hình `bnb_4bit_compute_dtype = torch.float16` giúp tận dụng tối đa nhân Tensor Cores của GPU T4 (tránh dùng `bfloat16` vì T4 không hỗ trợ phần cứng này natively).
*   **Tích lũy Gradient (Gradient Accumulation)**: Thiết lập kích thước lô thực tế bằng 1 và tích lũy qua 8 bước (`grad_accumulation_steps = 8`) để mô phỏng lô hiệu dụng bằng 8, giữ bộ nhớ kích hoạt (activation memory) ở mức tối thiểu.
*   **Khóa bảng nhúng chọn lọc (Embedding Gradient Hook)**: Chỉ cho phép cập nhật nhúng của 16 Latent Tokens mới, triệt tiêu gradient của các từ vựng gốc nhằm ngăn chặn hiện tượng trôi nghĩa từ vựng gốc (**Token Drift**).
*   **Mặt nạ phạt hữu hạn `-65500.0`**: Thay thế giá trị $-\infty$ để tăng tính ổn định số học trong tính toán Softmax của `float16`.

---

## 📂 3. Cấu Trúc Thư Mục Dự Án
```text
LIVR-Mini-Benchmark/
├── config/
│   ├── implement_config.json   # Cấu hình huấn luyện Stage 1 & Stage 2
│   └── evaluation_config.json  # Cấu hình đánh giá miền Novel Datasets
├── docs/
│   └── nghiem_thu.md           # Báo cáo nghiệm thu kỹ thuật chi tiết & ví dụ thực tế
├── src/
│   ├── model.py                # Quản lý nạp mô hình QLoRA, Vocab Expansion & Hook
│   ├── mask.py                 # Xây dựng custom attention mask & Monkey-patching
│   └── utils.py                # Pipeline tiền xử lý dữ liệu, pHash de-duplication
├── 01_implement_mini.ipynb     # Notebook huấn luyện & Đánh giá baseline (disable LoRA)
├── 02_evaluation_mini.ipynb    # Notebook adaptation 2 epochs & Sanity Check chặn ảnh
├── requirements.txt            # Danh sách thư viện bắt buộc
└── README.md                   # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🚀 4. Hướng Dẫn Cài Đặt & Sử Dụng

### Chạy Cục Bộ (Local Environment)
1. Khởi tạo môi trường ảo và cài đặt các thư viện phụ thuộc:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Trên Linux
   pip install -r requirements.txt
   ```
2. Cập nhật các siêu tham số trong các file cấu hình tại thư mục `config/`.
3. Khởi động Jupyter Lab và mở các file Notebook tương ứng để chạy thử nghiệm.

### Chạy Trên Google Colab (Khuyên dùng với T4 GPU)
1. Upload file Notebook `01_implement_mini.ipynb` lên Google Colab và đổi runtime sang **T4 GPU**.
2. Điền thông tin repository GitHub chứa mã nguồn nhánh `develop` của bạn vào Cell 1 để Notebook tự động clone và pull code mới nhất:
   ```python
   REPO_URL = "https://github.com/CodeDaoVietNam/LIVR-Mini-Benchmark.git"
   PROJECT_DIR = "LIVR-Mini-Benchmark"
   BRANCH = "develop"
   ```
3. Chạy tuần tự các cell tiếp theo. Checkpoints sẽ được đồng bộ trực tiếp lên Google Drive của bạn dưới đường dẫn `/content/drive/MyDrive/LIVR_Mini_Project/checkpoints/` để tránh mất dữ liệu khi bị timeout session.
4. Lặp lại tương tự với notebook `02_evaluation_mini.ipynb` để đánh giá trên Novel Datasets (`VisuLogic` và `CV-Bench`).

---

## 📈 5. Các Thử Nghiệm Kiểm Định Khoa Học
*   **Baseline Comparison**: Chạy thử nghiệm đối chiếu độ chính xác Accuracy giữa mô hình LIVR (có LoRA) và mô hình nguyên bản của hãng (ngắt LoRA thông qua context manager `model.disable_adapter()`).
*   **Sanity Check (Bịt mắt ảnh)**: Trong quá trình suy luận ở Giai đoạn đánh giá, mô hình sẽ bị chặn ảnh đột ngột (ép `model.livr_stage = 1`). Mức sụt giảm Accuracy nhỏ/vừa phải chứng minh các Latent Tokens đã đóng gói thành công tri thức thị giác.

---

## 📄 6. Tài Liệu Báo Cáo
Báo cáo nghiệm thu chi tiết, giải thích học thuật sâu sắc kèm ví dụ so sánh thực tế và giải pháp khắc phục các sự cố về bộ nhớ/lỗi thiết bị được lưu trữ đầy đủ tại: **[docs/nghiem_thu.md](docs/nghiem_thu.md)**.