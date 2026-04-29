import math

class RareFeatureScorer:
    def __init__(self, alpha=2.0, k=None, base_rate=0.5):
        """
        :param alpha: Smoothing factor (Bayes smoothing). Đóng vai trò là thông số shape của Beta Prior.
        :param k: (Deprecated) Tham số cũ đã bị loại bỏ, giữ lại trong signature để không break pipeline cũ.
        :param base_rate: Tỉ lệ malware của toàn bộ dataset (Mặc định 0.5 cho tập cân bằng).
        """
        self.alpha = alpha
        self.base_rate = base_rate

    def score_feature(self, col, stats_dict):
        """
        Tính toán điểm số cho các đặc trưng hiếm (Rare Features) dựa trên Toán học Bayes.
        Sử dụng Beta Distribution để đo lường Uncertainty (Độ bất định).
        """
        s = stats_dict.get(col, {})
        
        # Lấy thông số từ dictionary (Đã được dọn sẵn từ file Stats)
        malware_count = s.get("malware_count", 0)
        total_active_count = s.get("active_count", 0)

        # Xử lý an toàn: Nếu vì lý do nào đó JSON trả về None, tự động gán về 0
        if malware_count is None: malware_count = 0
        if total_active_count is None: total_active_count = 0

        m = float(malware_count)
        n = float(total_active_count)

        # Nếu tính năng không xuất hiện lần nào (hoặc cực kỳ vô dụng)
        if n == 0:
            return "DROP_RARE_LOW_SCORE", 0.0

        # ==========================================
        # SUPERVISED SCORING (BETA DISTRIBUTION)
        # ==========================================
        
        # 1. Expected Value (Mean của Beta distribution)
        # Đóng vai trò là xác suất đã được smoothed
        mean = (m + self.alpha) / (n + 2 * self.alpha)
        
        # 2. Variance (Độ bất định - Uncertainty)
        # n càng nhỏ -> Variance càng lớn -> Càng thiếu chắc chắn
        var = (mean * (1.0 - mean)) / (n + 2 * self.alpha + 1.0)
        std_dev = math.sqrt(var)
        
        # 3. Z-Score (Điểm số chuẩn hóa dựa trên độ bất định)
        # Lệch càng nhiều so với base_rate VÀ std_dev càng nhỏ -> Score càng cao
        score = abs(mean - self.base_rate) / (std_dev + 1e-9)  # Cộng 1e-9 tránh chia 0

        # 4. Decision Rule (Bộ quy tắc ra quyết định theo Z-score / Standard Deviation)
        # Điểm số giờ đây thể hiện: "Feature này lệch khỏi base_rate bao nhiêu lần độ lệch chuẩn?"
        if score >= 2.5:       # Lệch cực kỳ rõ ràng (Tương đương khoảng 99% confident)
            action = "KEEP_RARE_HIGH_SCORE"
        elif score >= 1.5:     # Có tín hiệu nhưng không quá áp đảo (Tương đương 85-90% confident)
            action = "TRANSFORM_RARE_MID_SCORE"
        else:                  # Tín hiệu yếu, có thể do nhiễu
            action = "DROP_RARE_LOW_SCORE"

        return action, score