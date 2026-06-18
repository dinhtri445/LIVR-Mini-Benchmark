# src/mask_kaggle.py
import torch
import types

def generate_stage1_bottleneck_mask(seq_len, img_start, img_end, prompt_end, latent_end, device="cpu"):
    """
    Xây dựng ma trận Custom Causal Attention Mask cho Stage 1 (Bottleneck Phase).
    Sửa lỗi Device Mismatch và tối ưu hóa tính toán số học bfloat16.
    """
    # 1. Khởi tạo ma trận tự hồi quy trên đúng thiết bị (device)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    
    # 2. Định cấu trúc các vùng chỉ số
    img_indices = torch.arange(img_start, img_end, device=device)
    prompt_indices = torch.arange(img_end, prompt_end, device=device)
    latent_indices = torch.arange(prompt_end, latent_end, device=device)
    answer_indices = torch.arange(latent_end, seq_len, device=device)
    
    # BỊT MẮT 1: Ép Answer tokens KHÔNG được nhìn trực tiếp Image tokens
    mask[answer_indices.unsqueeze(1), img_indices] = False
    
    # BỊT MẮT 2: Ép Prompt tokens KHÔNG được nhìn trực tiếp Image tokens
    mask[prompt_indices.unsqueeze(1), img_indices] = False
    
    # GIỮ NGUYÊN HOẶC MỞ RỘNG LUỒNG: 
    # Latent tokens nhìn Prompt: Mở luồng để hiểu ngữ cảnh câu hỏi
    mask[latent_indices.unsqueeze(1), prompt_indices] = True
    
    # 3. Tạo ma trận cộng thích hợp với float32
    float_mask = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
    # Dùng -30000.0 thay cho float("-inf") để tăng độ ổn định toán học trong float16
    float_mask = float_mask.masked_fill(~mask, -30000.0)
    
    return float_mask

def patch_model_for_livr(model, latent_token_ids, image_pad_token_id, pad_token_id):
    """
    Monkey-patch hàm forward của Qwen2.5-VL để tự động chèn ma trận mask (phiên bản tương thích Kaggle).
    """
    original_forward = model.forward
    
    def livr_forward(
        self,
        *args,
        **kwargs
    ):
        args = list(args)
        def get_arg(name, index, default=None):
            if name in kwargs:
                return kwargs[name]
            if len(args) > index:
                return args[index]
            return default

        def set_arg(name, index, value):
            if name in kwargs:
                kwargs[name] = value
            elif len(args) > index:
                args[index] = value
            else:
                kwargs[name] = value

        input_ids = get_arg("input_ids", 0)
        attention_mask = get_arg("attention_mask", 1)
        position_ids = get_arg("position_ids", 2)
        pixel_values = get_arg("pixel_values", 10)
        image_grid_thw = get_arg("image_grid_thw", 12)
        labels = get_arg("labels", 5)

        stage = getattr(self, "livr_stage", 1)
        
        # Chỉ can thiệp trong quá trình training (khi có labels)
        if labels is not None and input_ids is not None:
            # Xác định các mốc vị trí động dựa trên batch
            batch_size, seq_len = input_ids.size()
            device = input_ids.device
            
            # Vì Qwen2.5-VL có cơ chế xử lý song song hoặc độ dài thay đổi theo lô,
            # ta sinh mask cho từng mẫu trong batch rồi ghép lại
            custom_masks = []
            for b in range(batch_size):
                curr_ids = input_ids[b]
                curr_labels = labels[b]
                
                # Tìm mốc vị trí
                img_mask = (curr_ids == image_pad_token_id)
                img_positions = torch.where(img_mask)[0]
                
                if len(img_positions) == 0:
                    # Nếu không có ảnh, dùng causal mask mặc định
                    m = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
                    fm = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device).masked_fill(~m, -30000.0)
                    custom_masks.append(fm.unsqueeze(0))
                    continue
                    
                img_start = img_positions[0].item()
                img_end = img_positions[-1].item() + 1
                
                # Tìm vị trí kết thúc của latent tokens
                latent_positions = []
                for lid in latent_token_ids:
                    pos = torch.where(curr_ids == lid)[0]
                    if len(pos) > 0:
                        latent_positions.append(pos[0].item())
                
                if len(latent_positions) == 0:
                    m = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
                    fm = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device).masked_fill(~m, -30000.0)
                    custom_masks.append(fm.unsqueeze(0))
                    continue
                    
                prompt_end = min(latent_positions)
                latent_end = max(latent_positions) + 1
                
                # Sinh mask
                if stage == 1:
                    m = generate_stage1_bottleneck_mask(
                        seq_len=seq_len,
                        img_start=img_start,
                        img_end=img_end,
                        prompt_end=prompt_end,
                        latent_end=latent_end,
                        device=device
                    )
                else:
                    # Stage 2: Causal mask bình thường
                    m_bool = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
                    m = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device).masked_fill(~m_bool, -30000.0)
                    
                custom_masks.append(m.unsqueeze(0))
                
            # 1. Tự động tính toán position_ids bằng 2D attention_mask gốc trước khi ghi đè mask 4D
            # Tìm backbone model (Qwen2_5_VLModel) chứa phương thức get_rope_index
            if position_ids is None:
                backbone = self
                if hasattr(backbone, "model"):
                    backbone = backbone.model
                    if hasattr(backbone, "model") and not hasattr(backbone, "get_rope_index"):
                        backbone = backbone.model
                
                import inspect
                sig = inspect.signature(backbone.get_rope_index)
                rope_kwargs = {
                    "input_ids": input_ids,
                    "mm_token_type_ids": kwargs.get("mm_token_type_ids"),
                    "image_grid_thw": image_grid_thw,
                    "video_grid_thw": kwargs.get("video_grid_thw"),
                    "second_per_grid_ts": kwargs.get("second_per_grid_ts"),
                    "attention_mask": attention_mask,
                }
                # Lọc bỏ các tham số không được hỗ trợ bởi phiên bản transformers hiện tại
                rope_kwargs = {k: v for k, v in rope_kwargs.items() if k in sig.parameters}
                
                position_ids, mrope_position_ids = backbone.get_rope_index(**rope_kwargs)
                set_arg("position_ids", 2, position_ids)
                kwargs["mrope_position_ids"] = mrope_position_ids

            # 2. Đè lên trường attention_mask truyền vào transformer
            attention_mask = torch.stack(custom_masks, dim=0).to(dtype=self.dtype)
            set_arg("attention_mask", 1, attention_mask)
            
        return original_forward(*args, **kwargs)
        
    model.forward = types.MethodType(livr_forward, model)
    model.livr_stage = 1
    print("---> Đã tích hợp Custom Attention Mask tương thích Kaggle thành công!")
