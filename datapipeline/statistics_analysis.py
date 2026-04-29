import pandas as pd
import numpy as np
import json
from pathlib import Path

def run(train_csv_path, stats_json_path):
    print("  -> Tính toán Statistics (Min, Max, Quantiles, Spread)...")
    df_train = pd.read_csv(train_csv_path)
    
    # ==========================================
    # 1. KHAI THÁC NHÃN & THÔNG SỐ TỔNG QUAN (SUMMARY)
    # ==========================================
    total_rows = len(df_train)
    has_label = "label_L1" in df_train.columns
    
    if has_label:
        is_malware_series = (df_train["label_L1"] == "Malware")
        total_malware = int(is_malware_series.sum())
        base_rate = round(total_malware / total_rows, 4)
    else:
        is_malware_series = None
        total_malware = 0
        base_rate = 0.0
    
    # ==========================================
    # 2. CHỐNG RÒ RỈ NHÃN (Target Leakage)
    # ==========================================
    leakage_cols = ['label_L1', 'label_L2', 'label_L3', 'Class', 'Category']
    feature_cols = [c for c in df_train.columns if c not in leakage_cols]
    
    # Lấy các cột số từ danh sách đã lọc nhãn
    numerical_features = df_train[feature_cols].select_dtypes(include=[np.number])
    total_features = numerical_features.shape[1]
    
    print(f"  📊 Đang phân tích chay toàn bộ {total_features} cột (bao gồm cả cột hằng số).")
    
    stats_dict = {}
    
    # ==========================================
    # 3. TÍNH TOÁN CHI TIẾT TỪNG FEATURE
    # ==========================================
    for col in numerical_features.columns:
        series = numerical_features[col]
        
        # Tính phân vị
        q1 = float(series.quantile(0.01))
        q5 = float(series.quantile(0.05))
        q25 = float(series.quantile(0.25))
        q50 = float(series.quantile(0.50))
        q75 = float(series.quantile(0.75))
        q90 = float(series.quantile(0.90))
        q95 = float(series.quantile(0.95))
        q99 = float(series.quantile(0.99))
        col_min = float(series.min())
        col_max = float(series.max())
        
        # Tính Spread Metric
        spread_90 = q95 - q5
        if spread_90 < 1e-8:
            spread_90 = q75 - q25
            if spread_90 < 1e-8:
                spread_90 = 1e-6
                
        # Xử lý an toàn Skewness/Kurtosis
        skew_val = series.skew()
        kurt_val = series.kurtosis()
        
        # Đếm Active Count và Malware Count (CHO TRƯỜNG HỢP B BAYES)
        is_active = (series != 0)
        active_count = int(is_active.sum())
        malware_count = int((is_active & is_malware_series).sum()) if has_label else None
        
        # Đóng gói từng feature
        stats_dict[col] = {
            "data_type": str(series.dtype),
            "unique_values": int(series.nunique()),
            "missing_values": int(series.isnull().sum()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": col_min,
            "max": col_max,
            "skewness": float(0.0 if pd.isna(skew_val) else skew_val),
            "kurtosis": float(0.0 if pd.isna(kurt_val) else kurt_val),
            "zero_ratio": float((series == 0).mean()),
            "active_count": active_count,
            "malware_count": malware_count,
            "quantiles": {
                "1%": q1, "5%": q5, "25%": q25, "50%": q50,
                "75%": q75, "90%": q90, "95%": q95, "99%": q99
            },
            "spread_metric": float(spread_90)
        }

    # ==========================================
    # 4. ĐÓNG GÓI CHUNG (KÈM SUMMARY) & LƯU FILE
    # ==========================================
    final_output = {
        "_summary": {
            "total_rows_analyzed": total_rows,
            "total_features_analyzed": total_features,
            "total_malware_samples": total_malware,
            "dataset_base_rate": base_rate,
            "note": "Statistics extracted for Machine Learning pipeline. Contains Supervised metrics for Rare Feature Scoring."
        },
        "features": stats_dict
    }

    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"  ✅ Đã lưu {total_features} features (Kèm Summary) vào {stats_json_path}")

# ==========================================
# KHỐI CHẠY ĐỘC LẬP
# ==========================================
if __name__ == "__main__":
    FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent 
    DATA_DIR = PROJECT_ROOT / "data"

    print("🚀 CHẠY ĐỘC LẬP: STATISTICS ANALYSIS")
    
    relative_dirs = ["split_80_20"]

    for rel_dir in relative_dirs:
        base_dir = DATA_DIR / rel_dir
        dir_raw = base_dir / "raw"
        dir_metadata = base_dir / "metadata"
        dir_metadata.mkdir(parents=True, exist_ok=True)

        train_path = dir_raw / "train_raw.csv"
        stats_path = dir_metadata / "feature_stats.json"

        if train_path.exists():
            print(f"\n📂 ĐANG XỬ LÝ: {base_dir}")
            run(str(train_path), str(stats_path))
        else:
            print(f"⚠️ Bỏ qua {base_dir}: Không tìm thấy file train_raw.csv")