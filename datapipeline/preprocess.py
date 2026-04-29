import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

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

    # Load dữ liệu
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

    artifacts = {"clipping_bounds": {}, "group_scales": {}}
    
    # THÊM MỚI: Biến lưu trữ báo cáo chi tiết các phép toán đã áp dụng
    audit_report = {
        "pipeline_name": "Sensitive Preprocess Pipeline",
        "groups": {}
    }

    # ==========================================
    # STEP 1 — CLIPPING (Outer Fence 3*IQR)
    # ==========================================
    logging.info("   [1/3] Supervised Clipping (Outer Fences)...")
    for group_name, g in groups_summary.items():
        audit_report["groups"][group_name] = {
            "action_label": g["action"],
            "semantic_type": g.get("semantic_type", "generic"),
            "transform_math": "None",
            "scale_math": "None",
            "features_detail": {}
        }

        for col in g["features"].keys():
            if col not in X_train.columns: continue
            
            s = stats[col]
            q25 = s.get("quantiles", {}).get("25%", 0.0)
            q75 = s.get("quantiles", {}).get("75%", 0.0)
            iqr = q75 - q25
            
            lower_fence = q25 - 3 * iqr
            upper_fence = q75 + 3 * iqr
            
            q01 = s.get("quantiles", {}).get("1%", 0.0)
            q99 = s.get("quantiles", {}).get("99%", 0.0)
            lower_fence = min(lower_fence, q01)
            upper_fence = max(upper_fence, q99)

            X_train[col] = X_train[col].clip(lower=lower_fence, upper=upper_fence)
            X_val[col] = X_val[col].clip(lower=lower_fence, upper=upper_fence)
            
            artifacts["clipping_bounds"][col] = {"lower": lower_fence, "upper": upper_fence}
            
            # Ghi chép vào report
            audit_report["groups"][group_name]["features_detail"][col] = {
                "clipped_at": {"lower": round(lower_fence, 6), "upper": round(upper_fence, 6)}
            }

    # ==========================================
    # STEP 2 — TRANSFORM (Signed Square Root)
    # ==========================================
    logging.info("   [2/3] Transforming (Signed SQRT / Handling tails)...")
    for group_name, g in groups_summary.items():
        action = g["action"]
        cols = [c for c in g["features"].keys() if c in X_train.columns]
        if not cols: continue

        if action in ["TRANSFORM_LONG_TAIL_RIGHT", "TRANSFORM_LONG_TAIL_LEFT", "TRANSFORM_HEAVY_TAIL"]:
            X_train[cols] = np.sign(X_train[cols]) * np.sqrt(np.abs(X_train[cols]))
            X_val[cols] = np.sign(X_val[cols]) * np.sqrt(np.abs(X_val[cols]))
            
            # Ghi chép vào report
            audit_report["groups"][group_name]["transform_math"] = "Signed Square Root (np.sign(x) * sqrt(|x|))"
        elif action == "TRANSFORM_OUTLIER":
             audit_report["groups"][group_name]["transform_math"] = "Outlier Clipped Only (No non-linear transform)"

    # ==========================================
    # STEP 3 — SCALING (Group Max-Abs Scaling)
    # ==========================================
    logging.info("   [3/3] Scaling (Group-wise Max-Abs Strategy)...")
    for group_name, g in groups_summary.items():
        cols = [c for c in g["features"].keys() if c in X_train.columns]
        if not cols: continue
        
        if "BINARY" in g["action"] or "RARE" in g["action"]:
            audit_report["groups"][group_name]["scale_math"] = "Skipped (Binary/Rare feature)"
            continue

        max_val = np.abs(X_train[cols].values).max()
        scale = max(float(max_val), EPS)

        X_train[cols] = X_train[cols] / scale
        X_val[cols] = X_val[cols] / scale
        artifacts["group_scales"][group_name] = scale
        
        # Ghi chép vào report
        audit_report["groups"][group_name]["scale_math"] = f"Group Max-Abs Scaling (Divided by {round(scale, 6)})"

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

    # XUẤT FILE BÁO CÁO (AUDIT REPORT)
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