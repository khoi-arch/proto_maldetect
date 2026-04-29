import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler 

# ==========================================
# CẤU HÌNH LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

EPS = 1e-6

def run(train_csv_path, val_csv_path, dir_metadata, dir_processed, target_col):
    logging.info("-> Running SENSITIVE Preprocess Pipeline (Signal Preservation)...")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    with open(os.path.join(dir_metadata, "feature_stats.json"), 'r', encoding='utf-8') as f:
        stats = json.load(f)["features"]

    with open(os.path.join(dir_metadata, "feature_groups.json"), 'r', encoding='utf-8') as f:
        groups_metadata = json.load(f)
    groups_summary = groups_metadata["groups_summary"]

    leakage_cols = ['label_L1', 'label_L2', 'label_L3', 'Class', 'Category']
    X_train = df_train.drop(columns=leakage_cols, errors='ignore').copy()
    X_val = df_val.drop(columns=leakage_cols, errors='ignore').copy()

    valid_cols = []
    for g in groups_summary.values():
        valid_cols.extend(g["features"].keys())

    X_train = X_train[[c for c in X_train.columns if c in valid_cols]].copy()
    X_val = X_val[[c for c in X_val.columns if c in valid_cols]].copy()

    artifacts = {"clipping_bounds": {}, "scalers": {}}
    
    audit_report = {
        "pipeline_name": "Sensitive Preprocess Pipeline (Feature-wise Standard Scaled)",
        "groups": {}
    }

    # ==========================================
    # STEP 1 — CLIPPING (FIX: Dùng thẳng q01 và q99)
    # ==========================================
    logging.info("   [1/3] Hard Clipping (q01 - q99)...")
    for group_name, g in groups_summary.items():
        audit_report["groups"][group_name] = {
            "action_label": g["action"],
            "semantic_type": g.get("semantic_type", "generic"),
            "transform_math": "None",
            "scale_math": "Feature-wise StandardScaler (Mean=0, Std=1)",
            "features_detail": {}
        }

        for col in g["features"].keys():
            if col not in X_train.columns: continue
            
            s = stats[col]
            # Bỏ tính toán IQR cồng kềnh, dùng chặn 2 đầu cứng
            lower_fence = s.get("quantiles", {}).get("1%", 0.0)
            upper_fence = s.get("quantiles", {}).get("99%", 0.0)

            X_train[col] = X_train[col].clip(lower=lower_fence, upper=upper_fence)
            X_val[col] = X_val[col].clip(lower=lower_fence, upper=upper_fence)
            
            artifacts["clipping_bounds"][col] = {"lower": lower_fence, "upper": upper_fence}
            
            audit_report["groups"][group_name]["features_detail"][col] = {
                "clipped_at": {"lower": round(lower_fence, 6), "upper": round(upper_fence, 6)}
            }

    # ==========================================
    # STEP 2 — TRANSFORM (FIX: Dùng Signed Log1p thay vì SQRT)
    # ==========================================
    logging.info("   [2/3] Transforming (Signed Log1p)...")
    for group_name, g in groups_summary.items():
        action = g["action"]
        cols = [c for c in g["features"].keys() if c in X_train.columns]
        if not cols: continue

        if action in ["TRANSFORM_LONG_TAIL_RIGHT", "TRANSFORM_LONG_TAIL_LEFT", "TRANSFORM_HEAVY_TAIL"]:
            # Dùng np.log1p để xử lý mượt các giá trị gần 0 và giữ scale tốt cho số lớn
            X_train[cols] = np.sign(X_train[cols]) * np.log1p(np.abs(X_train[cols]))
            X_val[cols] = np.sign(X_val[cols]) * np.log1p(np.abs(X_val[cols]))
            audit_report["groups"][group_name]["transform_math"] = "Signed Log1p (np.sign(x) * np.log1p(abs(x)))"
        elif action == "TRANSFORM_OUTLIER":
             audit_report["groups"][group_name]["transform_math"] = "Outlier Clipped Only"

    # ==========================================
    # STEP 3 — SCALING (FIX: Scale mọi thứ trừ Binary)
    # ==========================================
    logging.info("   [3/3] Scaling (Feature-wise StandardScaler)...")
    
    # Kể cả biến RARE cũng phải scale để Transformer không bị nhiễu do chênh lệch độ lớn
    cols_to_scale = []
    for group_name, g in groups_summary.items():
        if "BINARY" not in g["action"]:  # Chỉnh sửa: Chỉ chừa BINARY ra, RARE vẫn đưa vào scale
            cols_to_scale.extend([c for c in g["features"].keys() if c in X_train.columns])
    
    if cols_to_scale:
        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
        artifacts["scalers"]["main_standard_scaler"] = scaler

    # ==========================================
    # SAVE & EXPORT
    # ==========================================
    final_cols = sorted(X_train.columns)
    X_train = X_train[final_cols]
    X_val = X_val[final_cols]
    
    X_train[target_col] = df_train[target_col].values
    X_val[target_col] = df_val[target_col].values

    os.makedirs(dir_processed, exist_ok=True)
    X_train.to_csv(os.path.join(dir_processed, "train_processed.csv"), index=False)
    X_val.to_csv(os.path.join(dir_processed, "val_processed.csv"), index=False)
    joblib.dump(artifacts, os.path.join(dir_metadata, "preprocess_artifacts.pkl"))

    report_path = os.path.join(dir_metadata, "preprocess_audit_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=4, ensure_ascii=False)

    logging.info(f"✅ Done. Data saved to {dir_processed}")
    logging.info(f"📄 Audit Report generated at: {report_path}")

if __name__ == "__main__":
    FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent 
    DATA_DIR = PROJECT_ROOT / "data"
    TARGET_COL = "label_L3"

    run(
        str(DATA_DIR / "split_80_20/raw/train_raw.csv"),
        str(DATA_DIR / "split_80_20/raw/val_raw.csv"),
        str(DATA_DIR / "split_80_20/metadata"),
        str(DATA_DIR / "split_80_20/processed"),
        TARGET_COL
    )