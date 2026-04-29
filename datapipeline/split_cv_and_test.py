import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN CHUẨN (PROJECT ROOT)
# ==========================================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "split_80_20"

RANDOM_SEED = 42
TEST_SIZE = 0.20          

def main():
    print(f"🚀 Project Root xác định tại: {PROJECT_ROOT}")
    print("🚀 Bắt đầu chia dữ liệu 80-20 Train/Test (Stratified)...")
   
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file gốc tại {RAW_DATA_PATH}")
   
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"📊 Đã tải dữ liệu gốc: {df_raw.shape[0]} samples.")
    
    # ==========================================
    # LOGIC TÁCH 3 LEVEL NHÃN (L1, L2, L3)
    # ==========================================
    print("⚠️ Đang chuẩn hóa và tạo 3 cấp độ nhãn (L1, L2, L3)...")
    
    clean_category = df_raw["Category"].astype(str).str.replace(r'\.raw$', '', regex=True)
    df_raw["label_L3"] = clean_category.apply(lambda x: "-".join(x.split("-")[:2]) if not x.startswith("Benign") else "Benign")
    df_raw["label_L2"] = clean_category.apply(lambda x: x.split("-")[0])
    df_raw["label_L1"] = clean_category.apply(lambda x: "Benign" if x.startswith("Benign") else "Malware")

    # ==========================================
    # CHỐNG RÒ RỈ NHÃN (Target Leakage)
    # ==========================================
    cols_to_drop = ["label_L1", "label_L2", "label_L3"]
    if "Category" in df_raw.columns: cols_to_drop.append("Category")
    if "Class" in df_raw.columns: cols_to_drop.append("Class")
        
    X = df_raw.drop(columns=cols_to_drop)
    y = df_raw[["label_L1", "label_L2", "label_L3"]]

    # ==========================================
    # CHIA 80-20
    # ==========================================
    raw_dir = DATA_OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y["label_L3"]
    )
    
    pd.concat([X_train, y_train], axis=1).to_csv(raw_dir / "train_raw.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(raw_dir / "val_raw.csv", index=False)
    print(f"🟢 Đã lưu tập Train (80%) và Val/Test (20%) tại: {raw_dir}")

    # ---------------------------------------------------------
    # 🔍 SANITY CHECKS & LOGGING
    # ---------------------------------------------------------
    overlap_count = len(set(X_train.index) & set(X_test.index))
    
    train_dist = y_train["label_L3"].value_counts(normalize=True) * 100
    test_dist = y_test["label_L3"].value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({
        "Train (%)": train_dist, 
        "Test (%)": test_dist
    }).round(2).fillna(0) 

    log_content = f"""==================================================
🔍 BÁO CÁO CHIA DỮ LIỆU (SANITY CHECKS)
==================================================
1. KÍCH THƯỚC (SIZE):
   - Tổng: {len(df_raw)} samples
   - Train (80%): {len(X_train)} samples
   - Test (20%): {len(X_test)} samples

2. GIAO THOA INDEX (ANTI-LEAK):
   - Trùng lặp Train/Test: {overlap_count}
   -> Trạng thái: {'PASS ✅' if overlap_count == 0 else 'FAILED ❌ (CÓ RÒ RỈ)'}

3. PHÂN PHỐI NHÃN L3 (STRATIFY CHECK):
{dist_df.to_string()}
==================================================
"""
    if overlap_count > 0:
        raise ValueError("🚨 PHÁT HIỆN RÒ RỈ DỮ LIỆU!")
    
    log_file_path = raw_dir / "split_sanity_report.txt"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    
    print(f"✅ Hoàn tất! Log file tại: {log_file_path}")

if __name__ == "__main__":
    main()