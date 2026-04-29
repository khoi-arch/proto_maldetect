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
# 2. KIẾN TRÚC FT-TRANSFORMER CHÍNH
# ==========================================
class FTTransformer(nn.Module):
    def __init__(self, num_features, num_classes, embed_dim=32, n_heads=4, n_layers=3, dropout=0.1, pooling_mode='mean'):
        """
        Mô hình Transformer dành cho dữ liệu bảng (Tabular Data).
        :param num_features: Số lượng cột features đầu vào.
        :param num_classes: Số lượng nhãn (classes) cần dự đoán.
        :param embed_dim: Kích thước vector nhúng (Khuyên dùng: 32, 64, 128).
        :param n_heads: Số lượng head trong Multi-Head Attention.
        :param n_layers: Số lớp Transformer Encoder.
        :param dropout: Tỷ lệ Dropout để chống overfit.
        :param pooling_mode: 'mean' (trung bình) hoặc 'cls' (dùng token đặc biệt).
        """
        super().__init__()
        self.pooling_mode = pooling_mode
        self.embed_dim = embed_dim
        
        # --- TOKENIZER & NORMALIZATION ---
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # --- POSITIONAL EMBEDDING (FIX 1) ---
        # Khởi tạo ma trận rỗng dư 1 thẻ cho CLS token (num_features + 1)
        self.feature_embeddings = nn.Parameter(torch.empty(1, num_features + 1, embed_dim))
        # Dùng trunc_normal_ với std=0.02 để tránh nhiễu ở giai đoạn đầu training
        nn.init.trunc_normal_(self.feature_embeddings, std=0.02)
        
        if self.pooling_mode == 'cls':
            self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # --- TRANSFORMER ENCODER (FIX 2) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu', # Dùng GELU mượt hơn ReLU cho Transformer
            batch_first=True,
            norm_first=True    # ĐỔI THÀNH PRE-NORM ĐỂ CHỐNG VANISHING GRADIENT
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # --- CLASSIFICATION HEAD ---
        self.ln = nn.LayerNorm(embed_dim)
        self.head_dropout = nn.Dropout(0.2) # Chốt chặn chống học vẹt ở đầu ra
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b = x.shape[0] # Batch size
        
        # Bước 1: Chuyển float thành vector & Chuẩn hóa
        x = self.tokenizer(x)
        x = self.token_norm(x)
        
        # Bước 2: Ghép Positional Embedding và xử lý CLS
        if self.pooling_mode == 'cls':
            # Mở rộng CLS token cho bằng với batch size và gắn vào đầu
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1) 
            
            # Cộng Positional Embedding (cho cả F features + 1 CLS)
            x = x + self.feature_embeddings
        else:
            # Chế độ Mean: Bỏ qua CLS, cộng Positional Embedding cho đúng F features
            x = x + self.feature_embeddings[:, :x.size(1), :]
            
        x = self.input_dropout(x)
        
        # Bước 3: Đi qua mạng Attention
        x = self.transformer(x)
        
        # Bước 4: Pooling lấy vector đại diện
        if self.pooling_mode == 'cls':
            out = x[:, 0] # Lấy vector đầu tiên (chính là CLS)
        else:
            out = x.mean(dim=1) # Lấy trung bình cộng của tất cả features
            
        # Bước 5: Phân loại
        out = self.ln(out)
        out = self.head_dropout(out)
        return self.head(out)


# ==========================================
# 3. TEST NHANH MÔ HÌNH (DÙNG ĐỂ DEBUG)
# ==========================================
if __name__ == "__main__":
    print("--- KIỂM TRA MÔ HÌNH FT-TRANSFORMER (BẢN TỐI ƯU SOTA) ---")
    
    # Giả lập tham số
    BATCH_SIZE = 64
    NUM_FEATURES = 50
    NUM_CLASSES = 3
    
    # Giả lập dữ liệu đầu vào (Giống hệt đầu ra của DataLoader)
    dummy_x = torch.randn(BATCH_SIZE, NUM_FEATURES)
    print(f"[*] Kích thước đầu vào (Input shape): {dummy_x.shape} -> [Batch_size, Num_features]")
    
    # Khởi tạo mô hình chế độ Mean Pooling
    model = FTTransformer(
        num_features=NUM_FEATURES, 
        num_classes=NUM_CLASSES, 
        embed_dim=32, 
        n_heads=4, 
        n_layers=3, 
        pooling_mode='mean'
    )
    
    # Tính toán thử
    output = model(dummy_x)
    
    print(f"[*] Kích thước đầu ra (Output shape) : {output.shape} -> [Batch_size, Num_classes]")
    print("[*] Trạng thái: ✅ Mô hình chạy thành công, không bị lỗi shape!")