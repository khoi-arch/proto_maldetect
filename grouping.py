import numpy as np
import json
import os
from pathlib import Path
from rare_scorer import RareFeatureScorer  # IMPORT SCORER MỚI

EPS = 1e-6

# ==========================================
# THÊM MỚI: HÀM NHẬN DIỆN SEMANTIC TYPE
# ==========================================
def get_semantic_type(col_name):
    col_lower = col_name.lower()
    if any(k in col_lower for k in ["count", "nproc", "num"]):
        return "count"
    elif any(k in col_lower for k in ["ratio", "pct", "percent", "rate"]):
        return "ratio"
    else:
        return "generic"

# ==========================================
# STEP 1: GÁN NHÃN XỬ LÝ (ACTION)
# ==========================================
def assign_action(s, total_rows=None):
    q01 = s.get("quantiles", {}).get("1%", 0.0)
    q50 = s.get("quantiles", {}).get("50%", 0.0)
    q99 = s.get("quantiles", {}).get("99%", 0.0)
    max_val = s.get("max", s.get("quantiles", {}).get("100%", q99))
    std = s.get("std", 0.0)  # Cần std để check outlier chuẩn xác
    
    real_spread = q99 - q01
    unique_vals = s.get("unique_values", 0)
    zero_ratio = s.get("zero_ratio", 0.0)
    skew = s.get("skewness", 0.0)
    kurt = s.get("kurtosis", 0.0)

    # ---------------------------------------------------------
    # 1. DROP OBVIOUS (Hiển nhiên)
    # ---------------------------------------------------------
    if unique_vals < 2:
        return "DROP_CONSTANT"

    if real_spread <= EPS and unique_vals > 50:
        return "DROP_MICRO_NOISE"

    # ---------------------------------------------------------
    # 2. BINARY CHECK (Ưu tiên trước RARE để tránh che lấp)
    # ---------------------------------------------------------
    if unique_vals == 2:
        minor_class_ratio = min(zero_ratio, 1.0 - zero_ratio)
        if minor_class_ratio > 0.001:
            return "KEEP_RAW_BINARY"

    # ---------------------------------------------------------
    # 3. RARE CHECK (CẬP NHẬT: Kết hợp count và ratio)
    # ---------------------------------------------------------
    rare_ratio = 1.0 - zero_ratio
    rare_count = total_rows * rare_ratio if total_rows is not None else float('inf')
    
    if rare_count < 50 or rare_ratio < 0.005:
        return "RARE_NEED_SCORING"

    # ---------------------------------------------------------
    # 4. TRANSFORM CHECK
    # ---------------------------------------------------------
    # Outlier áp đảo 
    median_safe = q50 if abs(q50) > EPS else EPS
    std_safe = std if std > EPS else (real_spread / 4.0 if real_spread > 0 else EPS)
    
    outlier_ratio_median = q99 / median_safe
    outlier_ratio_std = (q99 - q50) / std_safe

    if (abs(median_safe) > EPS and outlier_ratio_median > 20 and outlier_ratio_std > 5) or \
       (abs(q99) > EPS and (max_val / q99) > 10):
        return "TRANSFORM_OUTLIER"

    # Heavy tail (2 phía)
    if kurt > 5.0:  
        return "TRANSFORM_HEAVY_TAIL"

    # Long tail (đuôi dài)
    if skew > 1.0:
        return "TRANSFORM_LONG_TAIL_RIGHT"
    elif skew < -1.0:
        return "TRANSFORM_LONG_TAIL_LEFT"

    # ---------------------------------------------------------
    # 5. KEEP NORMAL
    # ---------------------------------------------------------
    return "KEEP_RAW_NORMAL"

# ==========================================
# STEP 3: GROUPING TRONG CÙNG 1 ACTION
# (CẬP NHẬT: Xử lý cả Spread Ratio và Max Ratio)
# ==========================================
def group_by_scale_and_max(features, stats_dict, base_multiplier=3, max_ratio_threshold=10.0):
    log_threshold = np.log10(base_multiplier)

    items = []
    for col in features:
        q01 = stats_dict[col].get("quantiles", {}).get("1%", 0.0)
        q99 = stats_dict[col].get("quantiles", {}).get("99%", 0.0)
        max_val = stats_dict[col].get("max", stats_dict[col].get("quantiles", {}).get("100%", q99))
        
        real_spread = q99 - q01
        spread = real_spread if real_spread > 0 else EPS
        scale = np.log10(spread + EPS)
        
        # Lấy độ lớn tuyệt đối của max để xét domination
        max_magnitude = abs(max_val) if abs(max_val) > EPS else EPS
        
        items.append((col, scale, max_magnitude))

    # Sort theo spread scale trước
    items.sort(key=lambda x: x[1])

    groups = []
    current_group = []
    current_min_scale = None
    current_max_scale = None
    current_min_max = None
    current_max_max = None

    for col, scale, max_mag in items:
        if current_min_scale is None:
            current_group = [(col, max_mag)]
            current_min_scale = scale
            current_max_scale = scale
            current_min_max = max_mag
            current_max_max = max_mag
            continue

        new_min_scale = min(current_min_scale, scale)
        new_max_scale = max(current_max_scale, scale)
        new_min_max = min(current_min_max, max_mag)
        new_max_max = max(current_max_max, max_mag)

        # Điều kiện 1: Chênh lệch log scale (Spread Ratio <= 3)
        spread_ok = (new_max_scale - new_min_scale) <= log_threshold
        # Điều kiện 2: Tránh Max Domination (Max Value Ratio <= 10)
        max_ratio_ok = (new_max_max / new_min_max) <= max_ratio_threshold

        if spread_ok and max_ratio_ok:
            current_group.append((col, max_mag))
            current_min_scale = new_min_scale
            current_max_scale = new_max_scale
            current_min_max = new_min_max
            current_max_max = new_max_max
        else:
            # Ngắt group, lưu group hiện tại và tạo group mới
            groups.append([c[0] for c in current_group])
            current_group = [(col, max_mag)]
            current_min_scale = scale
            current_max_scale = scale
            current_min_max = max_mag
            current_max_max = max_mag

    if current_group:
        groups.append([c[0] for c in current_group])

    return groups

# ==========================================
# MAIN RUN
# ==========================================
def run(stats_json_path, groups_json_path, base_multiplier=3):
    print("  -> Running Research-Grade Malware Feature Grouping...")

    if not os.path.exists(stats_json_path):
        print(f"❌ Missing {stats_json_path}")
        return

    with open(stats_json_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        
    stats_dict = full_data.get("features", {})
    total_rows = full_data.get("_summary", {}).get("total_rows_analyzed", None)

    # Khởi tạo Rare Scorer (Chỉ chạy Case B Bayes)
    rare_scorer = RareFeatureScorer(alpha=2.0, k=10.0, base_rate=0.5)

    dropped_cols = []
    rare_proofs = {}
    
    large_groups = {
        "KEEP_RAW_BINARY": [],
        "KEEP_RAW_NORMAL": [],
        "KEEP_RARE_HIGH_SCORE": [],
        "TRANSFORM_RARE_MID_SCORE": [],
        "TRANSFORM_OUTLIER": [],
        "TRANSFORM_LONG_TAIL_RIGHT": [],
        "TRANSFORM_LONG_TAIL_LEFT": [],
        "TRANSFORM_HEAVY_TAIL": []
    }

    # BƯỚC 1 & 2: Phân loại nhãn và Áp dụng Rare Scoring
    for col, s in stats_dict.items():
        action = assign_action(s, total_rows)
        
        if action == "RARE_NEED_SCORING":
            final_action, score = rare_scorer.score_feature(col, s)
            rare_proofs[col] = {"final_action": final_action, "score": round(score, 4)}
            action = final_action

        if action.startswith("DROP"):
            dropped_cols.append((col, action))
        else:
            if action not in large_groups:
                large_groups[action] = []
            large_groups[action].append(col)

    print(f"  🗑️ Dropped {len(dropped_cols)} invalid/noise features")
    print(f"  🔍 Processed {len(rare_proofs)} rare features through Scorer")

    # BƯỚC 3: Grouping theo Semantic Type, Độ rộng Spread và Max Domination
    final_groups = []
    group_id = 0

    for action, features in large_groups.items():
        if not features:
            continue

        # Tách feature theo Semantic Type trước khi xét scale
        semantic_buckets = {}
        for col in features:
            sem_type = get_semantic_type(col)
            if sem_type not in semantic_buckets:
                semantic_buckets[sem_type] = []
            semantic_buckets[sem_type].append(col)

        # Grouping trên từng sub-bucket (Action + Semantic Type)
        for sem_type, sem_features in semantic_buckets.items():
            subgroups = group_by_scale_and_max(sem_features, stats_dict, base_multiplier, max_ratio_threshold=10.0)

            for sub in subgroups:
                final_groups.append({
                    "name": f"Group_{group_id}",
                    "action": action,
                    "semantic_type": sem_type,
                    "features": sub
                })
                group_id += 1

    # BƯỚC 4: XUẤT METADATA & DOWNSTREAM EXECUTION PARAMETERS
    groups_summary = {}
    feature_to_group = {}

    for g in final_groups:
        g_name = g["name"]
        g_cols = g["features"]
        g_action = g["action"]
        g_semantic = g["semantic_type"]

        feature_proofs = {}
        group_spreads = []
        group_max_vals = []

        for col in g_cols:
            s = stats_dict[col]
            q01 = s.get("quantiles", {}).get("1%", 0.0)
            q99 = s.get("quantiles", {}).get("99%", 0.0)
            max_val = s.get("max", s.get("quantiles", {}).get("100%", q99))
            
            real_spread = q99 - q01
            spread = real_spread if real_spread > 0 else 0.0
            group_spreads.append(spread)
            
            max_mag = abs(max_val) if abs(max_val) > 0 else 0.0
            group_max_vals.append(max_mag)
            
            clip_bounds = {"lower": None, "upper": None}
            if g_action == "TRANSFORM_OUTLIER":
                clip_bounds["upper"] = q99
            elif g_action == "TRANSFORM_HEAVY_TAIL":
                clip_bounds["lower"] = q01
                clip_bounds["upper"] = q99
            
            proof_data = {
                "q1": q01,
                "q99": q99,
                "max": max_val,
                "spread": round(spread, 6),
                "skewness": round(s.get("skewness", 0.0), 4),
                "kurtosis": round(s.get("kurtosis", 0.0), 4),
                "unique_values": s.get("unique_values", 0),
                "clip_bounds": clip_bounds if "TRANSFORM" in g_action and (clip_bounds["lower"] is not None or clip_bounds["upper"] is not None) else None
            }
            
            if col in rare_proofs:
                proof_data["rare_score"] = rare_proofs[col]["score"]

            feature_proofs[col] = proof_data
            feature_to_group[col] = g_name

        min_spread = min(group_spreads) if group_spreads else 0
        max_spread = max(group_spreads) if group_spreads else 0
        spread_ratio = (max_spread / min_spread) if min_spread > 0 else 0

        min_max_val = min(group_max_vals) if group_max_vals else 0
        max_max_val = max(group_max_vals) if group_max_vals else 0
        max_value_ratio = (max_max_val / min_max_val) if min_max_val > 0 else 0

        groups_summary[g_name] = {
            "action": g_action,
            "semantic_type": g_semantic,
            "feature_count": len(g_cols),
            "proof_of_grouping": {
                "min_spread_in_group": round(min_spread, 6),
                "max_spread_in_group": round(max_spread, 6),
                "spread_ratio_max_vs_min": round(spread_ratio, 2),
                "max_value_ratio_max_vs_min": round(max_value_ratio, 2)
            },
            "features": feature_proofs
        }

    # Tracking lý do bị Drop
    dropped_proofs = {}
    for item in dropped_cols:
        col = item[0]
        drop_reason = item[1]
        
        s = stats_dict[col]
        q01 = s.get("quantiles", {}).get("1%", 0.0)
        q99 = s.get("quantiles", {}).get("99%", 0.0)
        real_spread = q99 - q01
        unique_vals = s.get("unique_values", 0)
        zero_ratio = s.get("zero_ratio", 0.0)
        
        rare_count = None
        if total_rows is not None:
            rare_count = int(total_rows * (1.0 - zero_ratio))

        drop_metadata = {
            "reason": drop_reason.replace("DROP_", "").lower(),
            "unique_values": unique_vals,
            "spread": round(real_spread, 6),
            "estimated_rare_events": rare_count
        }

        if col in rare_proofs:
            drop_metadata["rare_score"] = rare_proofs[col]["score"]

        dropped_proofs[col] = drop_metadata

    output = {
        "logic": "Research-Grade Malware Anomaly Detection V2 (Semantic Rules, Robust Outliers, Max-Domination Prevented)",
        "base_multiplier": base_multiplier,
        "total_rows_used": total_rows,
        "dropped_features_proof": dropped_proofs,
        "groups_summary": groups_summary,
        "feature_to_group": feature_to_group
    }

    with open(groups_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"  ✅ Done: {len(final_groups)} groups created")


# ==========================================
# EXECUTOR
# ==========================================
if __name__ == "__main__":
    FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent 
    DATA_DIR = PROJECT_ROOT / "data"

    BASE_MULTIPLIER = 3

    print("🚀 FINAL CORE-TAIL GROUPING PIPELINE V2")

    relative_dirs = ["split_80_20"]

    for rel_dir in relative_dirs:
        base_dir = DATA_DIR / rel_dir
        metadata_dir = base_dir / "metadata"

        stats_path = metadata_dir / "feature_stats.json"
        groups_path = metadata_dir / "feature_groups.json"

        os.makedirs(metadata_dir, exist_ok=True)

        if stats_path.exists():
            print(f"\n📂 Processing: {base_dir}")
            run(str(stats_path), str(groups_path), BASE_MULTIPLIER)
        else:
            print(f"⚠️ Skipped {base_dir} (missing stats at {stats_path})")