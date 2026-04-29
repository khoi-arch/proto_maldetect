import torch
import torch.nn as nn
import math

# ==========================================
# 1. BƯỚC NHÚNG (EMBEDDING) CHO SỐ THỰC
# ==========================================
class FeatureTokenizer(nn.Module):
    """
    Chuyển đổi mỗi giá trị số thực (float) của từng feature thành một vector (embedding).
    """
    def __init__(self, num_features, embed_dim):
        super().__init__()
        # Trọng số (weight) và độ lệch (bias) riêng cho từng feature
        self.weight = nn.Parameter(torch.Tensor(num_features, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_features, embed_dim))
        
        # Khởi tạo giá trị ngẫu nhiên chuẩn (Kaiming He)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Đầu vào x có shape: [Batch_size, Num_features]
        # Đầu ra có shape: [Batch_size, Num_features, Embed_dim]
        return x.unsqueeze(-1) * self.weight + self.bias


# ==========================================
# 2. KIẾN TRÚC HIERARCHICAL FT-TRANSFORMER
# ==========================================
class FTTransformer(nn.Module):
    def __init__(self, num_features, num_classes, embed_dim=128, n_heads=8, n_layers=4, dropout=0.1, pooling_mode='mean'):
        """
        Bản nâng cấp Hierarchical: Tách làm 2 chóp dự đoán để trị dứt điểm mất cân bằng.
        - embed_dim=128, n_heads=8, n_layers=4 để đủ sức chứa cho 15 loại mã độc.
        """
        super().__init__()
        self.pooling_mode = pooling_mode
        self.embed_dim = embed_dim
        
        # --- TOKENIZER & NORMALIZATION ---
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # --- POSITIONAL EMBEDDING ---
        # Khởi tạo ma trận rỗng dư 1 thẻ cho CLS token (num_features + 1)
        self.feature_embeddings = nn.Parameter(torch.empty(1, num_features + 1, embed_dim))
        nn.init.trunc_normal_(self.feature_embeddings, std=0.02)
        
        if self.pooling_mode == 'cls':
            self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # --- TRANSFORMER ENCODER ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu', 
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # --- CLASSIFICATION HEADS ---
        self.ln = nn.LayerNorm(embed_dim)
        self.head_dropout = nn.Dropout(0.2)
        
        # Nhánh 1: Dự đoán Binary (Benign = 0 vs Malware = 1)
        self.binary_head = nn.Linear(embed_dim, 2)
        
        # Nhánh 2: Dự đoán Family (15 loại Malware)
        # num_classes gốc là 16, trừ đi 1 (Benign) còn 15
        self.family_classes = num_classes - 1
        self.family_head = nn.Linear(embed_dim, self.family_classes)

    def forward(self, x):
        b = x.shape[0] 
        
        x = self.tokenizer(x)
        x = self.token_norm(x)
        
        if self.pooling_mode == 'cls':
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1) 
            x = x + self.feature_embeddings
        else:
            x = x + self.feature_embeddings[:, :x.size(1), :]
            
        x = self.input_dropout(x)
        x = self.transformer(x)
        
        if self.pooling_mode == 'cls':
            out = x[:, 0]
        else:
            out = x.mean(dim=1) 
            
        out = self.ln(out)
        out = self.head_dropout(out)
        
        # QUAN TRỌNG: Chỉ trả về vector đặc trưng (out), không gọi 2 cái head ở đây.
        # Việc gọi head sẽ được thực hiện thủ công trong train.py để cô lập gradient.
        return out


# ==========================================
# 3. TEST NHANH MÔ HÌNH (DÙNG ĐỂ DEBUG)
# ==========================================
if __name__ == "__main__":
    print("--- KIỂM TRA HIERARCHICAL FT-TRANSFORMER ---")
    
    BATCH_SIZE = 64
    NUM_FEATURES = 50
    NUM_CLASSES = 16 # 1 Benign + 15 Malware
    
    dummy_x = torch.randn(BATCH_SIZE, NUM_FEATURES)
    print(f"[*] Kích thước đầu vào: {dummy_x.shape}")
    
    model = FTTransformer(
        num_features=NUM_FEATURES, 
        num_classes=NUM_CLASSES, 
        embed_dim=128, 
        n_heads=8, 
        n_layers=4, 
        pooling_mode='mean'
    )
    
    # Model giờ trả về embeddings
    out = model(dummy_x)
    logits_bin = model.binary_head(out)
    logits_fam = model.family_head(out)
    
    print(f"[*] Đầu ra Vector Đặc trưng: {out.shape} -> [Batch, Embed_dim]")
    print(f"[*] Đầu ra Nhánh Binary: {logits_bin.shape} -> [Batch, 2]")
    print(f"[*] Đầu ra Nhánh Family: {logits_fam.shape} -> [Batch, 15]")
    print("[*] Trạng thái: ✅ Mô hình phân tầng chạy thành công!")