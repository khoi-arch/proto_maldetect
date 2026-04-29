import os
import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

class MalwareDataset(Dataset):
    def __init__(self, csv_path, target_col='label_L3', label_encoder=None, encoder_save_path=None, train_columns=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file tại: {csv_path}")

        df = pd.read_csv(csv_path)
        
        # 1. BẢO VỆ THỨ TỰ CỘT (FIX 2)
        X_df = df.drop(columns=[target_col])
        self.feature_names = X_df.columns.tolist()
        
        # Nếu là tập Val/Test, phải kiểm tra thứ tự cột có khớp với Train không
        if train_columns is not None:
            if self.feature_names != train_columns:
                raise ValueError("Thứ tự cột của tập Validation không khớp với tập Train. Nguy cơ hỏng model!")

        # Ép kiểu Features sang float32
        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        y_raw = df[target_col].values
        
        # 2. XỬ LÝ NHÃN & BẢO VỆ CRASH (FIX 3)
        if label_encoder is None:
            # Tập TRAIN
            self.label_encoder = LabelEncoder()
            encoded_y = self.label_encoder.fit_transform(y_raw)
            
            if encoder_save_path:
                os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
                joblib.dump(self.label_encoder, encoder_save_path)
        else:
            # Tập VAL/TEST
            self.label_encoder = label_encoder
            
            # Kiểm tra xem có nhãn nào mới toanh chưa từng xuất hiện ở Train không
            unknown_mask = ~np.isin(y_raw, self.label_encoder.classes_)
            if unknown_mask.any():
                unseen_labels = np.unique(y_raw[unknown_mask])
                raise ValueError(f"Crash Alert: Tập Val chứa nhãn chưa từng thấy ở Train: {unseen_labels}")
                
            encoded_y = self.label_encoder.transform(y_raw)
            
        self.y = torch.tensor(encoded_y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Đổi default batch_size lên 256 (FIX 5)
def get_dataloaders(train_csv, val_csv, target_col, metadata_dir, batch_size=256):
    encoder_path = os.path.join(metadata_dir, "label_encoder.pkl")

    # Khởi tạo Train Dataset
    train_ds = MalwareDataset(
        csv_path=train_csv,
        target_col=target_col,
        encoder_save_path=encoder_path
    )

    # Khởi tạo Val Dataset (Truyền encoder và feature_names từ Train sang để check)
    val_ds = MalwareDataset(
        csv_path=val_csv,
        target_col=target_col,
        label_encoder=train_ds.label_encoder,
        train_columns=train_ds.feature_names
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    return train_loader, val_loader, train_ds.label_encoder, train_ds.feature_names

if __name__ == "__main__":
    FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent 
    PROCESSED_DIR = PROJECT_ROOT / "data" / "split_80_20" / "processed"
    METADATA_DIR = PROJECT_ROOT / "data" / "split_80_20" / "metadata"
    TARGET_COL = "label_L3"

    train_csv = str(PROCESSED_DIR / "train_processed.csv")
    val_csv = str(PROCESSED_DIR / "val_processed.csv")

    train_loader, val_loader, encoder, features = get_dataloaders(
        train_csv, val_csv, TARGET_COL, str(METADATA_DIR), batch_size=256
    )

    print(f"[INFO] Data Loaders Ready!")
    print(f"       - Num features: {len(features)}")
    print(f"       - Train batches: {len(train_loader)} (Batch size: 256)")
    print(f"       - Val batches: {len(val_loader)}")