import os
import torch
from pathlib import Path

# --- CẬP NHẬT IMPORT THEO ĐÚNG CẤU TRÚC THƯ MỤC ---
from datapipeline.data_loader import get_dataloaders
from model import FTTransformer
from train import train_model

def main():
    print("🚀 KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN MALWARE DETECTOR")

    # ==========================================
    # 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
    # ==========================================
    # Lấy thư mục gốc (Nơi chứa file main.py)
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # Trỏ chính xác vào các thư mục theo tree bạn cung cấp
    PROCESSED_DIR = PROJECT_ROOT / "data" / "split_80_20" / "processed"
    METADATA_DIR = PROJECT_ROOT / "data" / "split_80_20" / "metadata"

    train_csv = str(PROCESSED_DIR / "train_processed.csv")
    val_csv = str(PROCESSED_DIR / "val_processed.csv")
    TARGET_COL = "label_L3" 

    # Tự động chọn GPU (CUDA/MPS) nếu có, không thì chạy CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện: {device}")

    # ==========================================
    # 2. TẢI DỮ LIỆU
    # ==========================================
    print("\n[*] Đang khởi tạo DataLoader...")
    train_loader, val_loader, label_encoder, feature_names = get_dataloaders(
        train_csv, val_csv, TARGET_COL, str(METADATA_DIR), batch_size=256
    )

    num_features = len(feature_names)
    num_classes = len(label_encoder.classes_)

    print(f"    - Số lượng features: {num_features}")
    print(f"    - Số lượng classes: {num_classes}")
    print(f"    - Nhãn mục tiêu: {label_encoder.classes_}")

    # Trích xuất train_labels để truyền vào hàm tính Class Weights
    print("[*] Đang tính toán phân phối nhãn (Class Distribution)...")
    train_labels = train_loader.dataset.y.numpy()

    # ==========================================
    # 3. KHỞI TẠO MÔ HÌNH
    # ==========================================
    print("\n[*] Đang khởi tạo mô hình FT-Transformer SOTA...")
    model = FTTransformer(
        num_features=num_features,
        num_classes=num_classes,
        embed_dim=32,      # Có thể điều chỉnh: 32, 64
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        pooling_mode='mean'
    )

# ==========================================
    # 4. BẮT ĐẦU HUẤN LUYỆN
    # ==========================================
    print("\n" + "="*50)
    print("🔥 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN 🔥")
    print("="*50)
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_labels=train_labels,
        label_names=label_encoder.classes_,  # 👈 DÒNG NÀY LÀ THỦ PHẠM GÂY LỖI NẾU THIẾU
        device=device,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5
    )

if __name__ == "__main__":
    main()