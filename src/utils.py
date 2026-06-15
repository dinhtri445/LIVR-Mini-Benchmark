import torch 
from datasets import load_dataset
import os
from PIL import Image
import numpy as np
import imagehash 
import copy

def load_and_inspect_livr_dataset():
    """
    Hàm kết nối API Hugging Face, nạp tập dữ liệu hỗn hợp đa tác vụ LIVR 
    và in báo cáo cấu trúc để nghiệm thu Tuần 1.
    """
    print("======TIẾN HÀNH KẾT NỐI VÀ TẢI DATASET TRÊN HUGGING FACE ======")
    
    # Tải dataset trực tiếp từ Link: https://huggingface.co/datasets/Kkuntal990/LIVR_mixed
    dataset = load_dataset("Kkuntal990/LIVR_mixed")
    print("\n[SUCCESS] Đã tải thành công Dataset!")
    print(f"Cấu trúc phân vùng hệ thống (Splits): \n{dataset}")
    sample_data = dataset['train'][0]
    print("\n====== MẪU KIỂM TRA THỬ NGHIỆM DÒNG ĐẦU TIÊN (SAMPLE INGESTION) ======")
    for key, value in sample_data.items():
        # Nếu là trường ảnh hoặc dữ liệu quá dài, chỉ in ra loại định dạng (type) để tránh tràn màn hình log
        if key == 'image' or len(str(value)) > 200:
            print(f"- Khóa '{key}': Định dạng dữ liệu -> {type(value)}")
        else:
            print(f"- Khóa '{key}': Giá trị thực tế -> {value}")
            
    return dataset

def filter_and_deduplicate_pipeline(dataset):
    """
    Giao thức tiền xử lý nâng cao (Tuần 2):
    1. Lọc dải đối tượng đếm từ 2 đến 10.
    2. Khử trùng lặp ảnh bằng thuật toán Perceptual Hashing (pHash).
    3. Định dạng cấu trúc Chat Template chuẩn cho Qwen2.5-VL.
    """
    print("\n====== BẮT ĐẦU CHẠY PIPELINE TIỀN XỬ LÝ NÂNG CAO ======")
    
    seen_hashes = set()
    cleaned_data = []
    
    # Duyệt qua tập train thô để tiến hành gạn lọc
    for item in dataset['train']:
        task_type = item.get('task', '')
        answer = str(item.get('answer', '')).strip()
        image_obj = item.get('image') # Đây là một đối tượng PIL Image do HF Datasets tự động load
        query = item.get('query', '')
        
        # --- BƯỚC 1: LỌC RIÊNG CHO TÁC VỤ COUNTING (Theo Spec tác giả) ---
        if task_type == 'counting':
            if not answer.isdigit() or not (2 <= int(answer) <= 10):
                continue # Bỏ qua nếu đối tượng đếm nằm ngoài dải 2-10
                
        # --- BƯỚC 2: KHỬ TRÙNG LẶP ẢNH (VISUAL DE-DUPLICATION) ---
        try:
            # Tính toán mã băm nhận thức (Perceptual Hash) của bức ảnh
            v_hash = imagehash.phash(image_obj)
            if v_hash in seen_hashes:
                continue # Bỏ qua nếu ảnh này đã từng xuất hiện (Trùng lặp trực quan)
            seen_hashes.add(v_hash)
        except Exception as e:
            continue # Bỏ qua nếu file ảnh bị lỗi cấu trúc vật lý
            
        # --- BƯỚC 3: ĐỊNH DẠNG CHAT TEMPLATE CHUẨN CHO QWEN2.5-VL ---
        # Chuyển đổi định dạng câu hỏi thô thành cấu trúc hội thoại đa phương thức
        formatted_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
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
        
        # Lưu lại bản ghi đã làm sạch và định dạng chuẩn
        cleaned_data.append({
            "conversation": formatted_conversation,
            "task": task_type
        })
        
    print(f"[SUCCESS] Quy trình kết thúc!")
    print(f"- Số lượng ảnh trùng lặp hoặc lỗi bị loại bỏ: {len(dataset['train']) - len(cleaned_data)}")
    print(f"- Tổng số mẫu 'sạch' đạt chuẩn giữ lại cho bản Mini: {len(cleaned_data)} mẫu.")
    
    return cleaned_data

def prepare_vqa_inputs(processor, conversation, latent_tokens, device="cuda"):
    """
    Chuẩn bị Tensor đầu vào cho Qwen2.5-VL (input_ids, attention_mask, pixel_values, labels).
    - Chèn Latent Tokens vào sau text prompt của User.
    - Tokenize toàn bộ cuộc hội thoại bằng processor.apply_chat_template.
    - Đặt nhãn labels = -100 cho toàn bộ các token trước phần trả lời của assistant.
    """
    # 1. Sao chép sâu cuộc hội thoại để tránh ảnh hưởng dữ liệu gốc
    conv = copy.deepcopy(conversation)
    
    # 2. Tìm phần text prompt của User trong cuộc hội thoại để chèn Latent Tokens
    latent_str = "".join(latent_tokens)
    for msg in conv:
        if msg["role"] == "user":
            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    # Chèn Latent Tokens vào cuối prompt câu hỏi
                    content_item["text"] = f"{content_item['text'].strip()}\n{latent_str}"
                    
    # 3. Áp dụng Chat Template cho cuộc hội thoại đầy đủ (để sinh input_ids đầy đủ)
    is_training = (conv[-1]["role"] == "assistant")
    full_text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=not is_training)
    
    # 4. Áp dụng Chat Template cho phần Prompt của User (để tìm độ dài prompt)
    user_conv = [msg for msg in conv if msg["role"] == "user"]
    prompt_text = processor.apply_chat_template(user_conv, tokenize=False, add_generation_prompt=True)
    
    # 5. Extract images
    images = []
    for msg in conv:
        if msg["role"] == "user":
            for content_item in msg["content"]:
                if content_item["type"] == "image":
                    images.append(content_item["image"])
                    
    # 6. Chạy qua processor (Đã sửa lỗi cấu trúc bọc mảng lồng nhau cho Batch kích thước 1)
    full_inputs = processor(text=[full_text], images=[images], padding=True, return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text], images=[images], padding=True, return_tensors="pt")
    
    # 7. Đẩy các Tensor lên GPU
    inputs = {k: v.to(device) for k, v in full_inputs.items()}
    
    # 8. Tạo nhãn labels: set non-assistant prompt token labels to -100
    labels = inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].size(1)
    labels[:, :prompt_len] = -100
    
    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Chạy thử độc lập file script để test hạ tầng mạng và pipeline làm sạch
    dataset = load_and_inspect_livr_dataset()
    filter_and_deduplicate_pipeline(dataset)
