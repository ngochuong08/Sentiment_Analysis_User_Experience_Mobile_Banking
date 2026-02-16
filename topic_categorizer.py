# -*- coding: utf-8 -*-
"""
TOPIC CATEGORIZER: Phân loại chủ đề reviews ngân hàng di động
================================================================
Keyword-based topic classification cho reviews tiếng Việt
- 10 chủ đề chính liên quan đến mobile banking
- Hỗ trợ multi-label (1 review có thể thuộc nhiều chủ đề)
- Tích hợp với pipeline ML và dashboard

Author: KLTN - Sentiment Analysis Mobile Banking
"""

import re
import pandas as pd
from collections import Counter


class BankingTopicCategorizer:
    """
    Phân loại chủ đề cho reviews ứng dụng ngân hàng di động.

    Sử dụng keyword-based matching với 10 chủ đề banking-specific:
    1. Giao dịch & Chuyển tiền
    2. Đăng nhập & Xác thực
    3. Giao diện & Trải nghiệm (UX)
    4. Hiệu suất & Lỗi kỹ thuật
    5. Chăm sóc khách hàng (CSKH)
    6. Bảo mật & An toàn
    7. Tính năng & Dịch vụ
    8. Cập nhật & Phiên bản
    9. Phí & Lãi suất
    10. Thông báo & Quảng cáo
    """

    # ============================================================
    # TOPIC DEFINITIONS WITH KEYWORDS
    # ============================================================
    TOPICS = {
        "Giao dịch & Chuyển tiền": {
            "keywords": [
                # Chuyển khoản
                "chuyển tiền",
                "chuyển khoản",
                "chuyển khoản",
                "ck ",
                "giao dịch",
                "chuyển",
                "gửi tiền",
                "nhận tiền",
                "chuyển nhanh",
                "chuyển chậm",
                # Thanh toán
                "thanh toán",
                "thanh toan",
                "tt ",
                "trả tiền",
                "trả bill",
                "hóa đơn",
                "hoa don",
                "điện nước",
                "tiền điện",
                "tiền nước",
                # Tiền & Số dư
                "số dư",
                "so du",
                "tài khoản",
                "tai khoan",
                "tk ",
                "nạp tiền",
                "rút tiền",
                "tiền",
                "biên lai",
                "biên nhận",
                # Giao dịch cụ thể
                "liên ngân hàng",
                "napas",
                "nội bộ",
                "qua ngân hàng",
                "trừ tiền",
                "cộng tiền",
                "hoàn tiền",
                "nhận được tiền",
                "tiền không đến",
                "mất tiền",
                "thiếu tiền",
                "dư tiền",
            ],
            "icon": "💰",
            "color": "#2563eb",
        },
        "Đăng nhập & Xác thực": {
            "keywords": [
                # Đăng nhập
                "đăng nhập",
                "dang nhap",
                "login",
                "log in",
                "đăng xuất",
                "vào app",
                "mở app",
                "vào ứng dụng",
                "không vào được",
                "không đăng nhập",
                "ko đăng nhập",
                "k đăng nhập",
                # Mật khẩu
                "mật khẩu",
                "mat khau",
                "password",
                "mk ",
                "quên mật khẩu",
                "đổi mật khẩu",
                "mã pin",
                "pin ",
                # OTP
                "otp",
                "mã xác thực",
                "mã xác nhận",
                "xác thực",
                "xác nhận",
                "mã giao dịch",
                "smart otp",
                "soft otp",
                # Sinh trắc học
                "vân tay",
                "van tay",
                "face id",
                "faceid",
                "quét mặt",
                "quet mat",
                "sinh trắc",
                "sinh trac",
                "nhận diện khuôn mặt",
                "touch id",
                "khuôn mặt",
                "khuon mat",
                # eKYC
                "ekyc",
                "kyc",
                "cccd",
                "căn cước",
                "can cuoc",
                "cmnd",
                "chip",
                "nfc",
                "quét chip",
                "xác minh danh tính",
                "xác minh",
                "định danh",
            ],
            "icon": "🔐",
            "color": "#7c3aed",
        },
        "Giao diện & Trải nghiệm": {
            "keywords": [
                # Giao diện
                "giao diện",
                "giao dien",
                "gd ",
                "ui ",
                "ux ",
                "thiết kế",
                "thiet ke",
                "design",
                "layout",
                "bố cục",
                # Trực quan
                "đẹp",
                "xấu",
                "xấu xí",
                "bắt mắt",
                "trực quan",
                "nhìn",
                "màu sắc",
                "mau sac",
                "font",
                "icon",
                "hình ảnh",
                # Trải nghiệm
                "dễ dùng",
                "dễ sử dụng",
                "de dung",
                "khó dùng",
                "khó sử dụng",
                "dễ thao tác",
                "khó thao tác",
                "phức tạp",
                "đơn giản",
                "tiện lợi",
                "tiện ích",
                "thuận tiện",
                "bất tiện",
                "trải nghiệm",
                "trai nghiem",
                "sử dụng",
                "thao tác",
                "hiển thị",
                "hien thi",
                "rối mắt",
                "rối",
                "lộn xộn",
                "thân thiện",
                "trình bày",
            ],
            "icon": "🎨",
            "color": "#059669",
        },
        "Hiệu suất & Lỗi kỹ thuật": {
            "keywords": [
                # Lag/Chậm
                "lag",
                "giật",
                "chậm",
                "cham ",
                "nặng",
                "châm",
                "load",
                "loading",
                "tải chậm",
                "mở chậm",
                "chậm chạp",
                "chập chờn",
                # Lỗi
                "lỗi",
                "loi ",
                "bug",
                "error",
                "lỗi kỹ thuật",
                "trục trặc",
                "hỏng",
                "hong ",
                "lỗi liên tục",
                "lỗi hoài",
                "lỗi miết",
                # Crash/Sập
                "crash",
                "sập",
                "sap ",
                "văng",
                "vang ",
                "out",
                "đóng",
                "tự tắt",
                "tự đóng",
                "sập app",
                "sập ứng dụng",
                "thoát ứng dụng",
                "thoát app",
                # Đơ/Treo
                "đơ",
                "treo",
                "đứng hình",
                "đứng máy",
                "không phản hồi",
                "không tải được",
                "không load",
                "quay vòng vòng",
                "xoay vòng",
                "loading mãi",
                # Tương thích
                "không tương thích",
                "tương thích",
                "phiên bản android",
                "phiên bản ios",
                "máy cũ",
                "đt cũ",
                "điện thoại cũ",
            ],
            "icon": "⚡",
            "color": "#dc2626",
        },
        "Chăm sóc khách hàng": {
            "keywords": [
                # CSKH
                "cskh",
                "chăm sóc khách hàng",
                "cham soc",
                "khách hàng",
                "hỗ trợ",
                "ho tro",
                "support",
                "trợ giúp",
                "giúp đỡ",
                # Tổng đài
                "tổng đài",
                "tong dai",
                "hotline",
                "gọi điện",
                "liên hệ",
                "lien he",
                "điện thoại",
                "số điện thoại",
                # Nhân viên
                "nhân viên",
                "nhan vien",
                "nv ",
                "tư vấn",
                "tu van",
                "tư vấn viên",
                "nhân viên hỗ trợ",
                "staff",
                # Phản hồi
                "phản hồi",
                "phan hoi",
                "feedback",
                "rep ",
                "reply",
                "trả lời",
                "tra loi",
                "phúc đáp",
                "giải quyết",
                "giai quyet",
                "khiếu nại",
                "khieu nai",
                "than phiền",
                "góp ý",
                # Thái độ
                "nhiệt tình",
                "nhiet tinh",
                "thái độ",
                "thái đô",
                "lịch sự",
                "vô trách nhiệm",
                "thiếu trách nhiệm",
                "không quan tâm",
                "phục vụ",
                "dịch vụ",
                "dich vu",
                "service",
            ],
            "icon": "📞",
            "color": "#ea580c",
        },
        "Bảo mật & An toàn": {
            "keywords": [
                # Bảo mật
                "bảo mật",
                "bao mat",
                "an toàn",
                "an toan",
                "security",
                "bảo vệ",
                "bao ve",
                "riêng tư",
                "privacy",
                # Lừa đảo
                "lừa đảo",
                "lua dao",
                "lừa",
                "đảo",
                "scam",
                "hack",
                "hacker",
                "tấn công",
                "chiếm",
                "chiếm đoạt",
                "đánh cắp",
                "tài khoản bị",
                "mất tài khoản",
                # Khóa
                "khóa tài khoản",
                "khóa app",
                "khóa thẻ",
                "khoá",
                "chặn",
                "block",
                "khóa",
                "giới hạn",
                # Mã hóa
                "mã hóa",
                "ma hoa",
                "ssl",
                "mã bảo mật",
                "token",
            ],
            "icon": "🛡️",
            "color": "#0891b2",
        },
        "Tính năng & Dịch vụ": {
            "keywords": [
                # QR
                "qr",
                "qr code",
                "quét qr",
                "quet qr",
                "mã qr",
                "vnpay qr",
                "scan",
                "quét mã",
                # Thẻ
                "thẻ",
                "the ",
                "visa",
                "mastercard",
                "jcb",
                "napas",
                "thẻ tín dụng",
                "thẻ ghi nợ",
                "thẻ atm",
                "mở thẻ",
                # Tiết kiệm & Vay
                "tiết kiệm",
                "tiet kiem",
                "gửi tiết kiệm",
                "lãi suất tiết kiệm",
                "vay",
                "vay tiền",
                "khoản vay",
                "trả góp",
                "tra gop",
                # Tính năng khác
                "tính năng",
                "tinh nang",
                "chức năng",
                "chuc nang",
                "tích hợp",
                "tich hop",
                "liên kết",
                "lien ket",
                "ví điện tử",
                "vi dien tu",
                "e-wallet",
                "ewallet",
                "bảo hiểm",
                "bao hiem",
                "đầu tư",
                "dau tu",
                "vpbank neo",
                "sacombank pay",
                "bidv smartbanking",
                # Chuyển đổi số
                "online",
                "trực tuyến",
                "số hóa",
                "digital",
            ],
            "icon": "⚙️",
            "color": "#4f46e5",
        },
        "Cập nhật & Phiên bản": {
            "keywords": [
                # Cập nhật
                "cập nhật",
                "cap nhat",
                "update",
                "nâng cấp",
                "nang cap",
                "upgrade",
                "bản mới",
                "phiên bản",
                "phien ban",
                "version",
                "bản cập nhật",
                "sau cập nhật",
                "sau khi cập nhật",
                "cập nhật mới",
                "cập nhật gần đây",
                "cập nhật xong",
                "trước cập nhật",
                "chưa cập nhật",
                "bắt cập nhật",
                "yêu cầu cập nhật",
                "cần cập nhật",
                # Sửa lỗi
                "sửa lỗi",
                "sua loi",
                "fix",
                "vá lỗi",
                "patch",
                "khắc phục",
                "khac phuc",
                "cải thiện",
                "cai thien",
                # Phiên bản cụ thể
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "version",
                "ios 17",
                "ios 18",
                "android 13",
                "android 14",
            ],
            "icon": "🔄",
            "color": "#0d9488",
        },
        "Phí & Lãi suất": {
            "keywords": [
                # Phí
                "phí",
                "phi ",
                "lệ phí",
                "le phi",
                "phí dịch vụ",
                "phí giao dịch",
                "phí chuyển tiền",
                "phí chuyển khoản",
                "phí duy trì",
                "phí thường niên",
                "phí hàng tháng",
                "phí hàng năm",
                "phí sms",
                "phí tin nhắn",
                # Miễn phí
                "miễn phí",
                "mien phi",
                "free",
                "không mất phí",
                "không tính phí",
                "phí 0",
                "zero phí",
                # Lãi suất
                "lãi suất",
                "lai suat",
                "lãi",
                "interest",
                "rate",
                "lãi cao",
                "lãi thấp",
                "lãi tốt",
                "lãi hấp dẫn",
                # Trừ phí
                "trừ phí",
                "tru phi",
                "thu phí",
                "tính phí",
                "charge",
                "đắt",
                "rẻ",
                "giá",
                "chi phí",
                "tốn",
                "trừ tiền phí",
                "mất phí",
            ],
            "icon": "💸",
            "color": "#ca8a04",
        },
        "Thông báo & Quảng cáo": {
            "keywords": [
                # Thông báo
                "thông báo",
                "thong bao",
                "notification",
                "noti",
                "push",
                "nhắc nhở",
                "nhac nho",
                "cảnh báo",
                "canh bao",
                "biến động số dư",
                "bdsd",
                "biến động",
                "bien dong",
                "thông báo giao dịch",
                "thông báo chuyển tiền",
                "tin nhắn",
                "sms",
                "message",
                # Quảng cáo
                "quảng cáo",
                "quang cao",
                "ads",
                "ad ",
                "advertising",
                "pop up",
                "popup",
                "pop-up",
                "banner",
                "spam",
                "khuyến mãi",
                "khuyen mai",
                "ưu đãi",
                "uu dai",
                "voucher",
                "coupon",
                "giảm giá",
                "giam gia",
                "promotion",
                "promo",
            ],
            "icon": "📢",
            "color": "#9333ea",
        },
    }

    # Topic display order
    TOPIC_ORDER = [
        "Giao dịch & Chuyển tiền",
        "Đăng nhập & Xác thực",
        "Giao diện & Trải nghiệm",
        "Hiệu suất & Lỗi kỹ thuật",
        "Chăm sóc khách hàng",
        "Bảo mật & An toàn",
        "Tính năng & Dịch vụ",
        "Cập nhật & Phiên bản",
        "Phí & Lãi suất",
        "Thông báo & Quảng cáo",
    ]

    def __init__(self):
        """Khởi tạo categorizer và compile regex patterns cho hiệu suất."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns cho từng chủ đề."""
        self.compiled_patterns = {}
        for topic, config in self.TOPICS.items():
            # Build regex pattern: word boundary matching cho từng keyword
            patterns = []
            for kw in config["keywords"]:
                # Escape special regex characters
                escaped = re.escape(kw.strip())
                patterns.append(escaped)

            # Sort by length (longest first) for greedy matching
            patterns.sort(key=len, reverse=True)

            # Combine into single regex with word boundaries where appropriate
            combined = "|".join(patterns)
            self.compiled_patterns[topic] = re.compile(
                f"(?:{combined})", re.IGNORECASE | re.UNICODE
            )

    def categorize_single(self, text: str) -> list:
        """
        Phân loại chủ đề cho 1 review.

        Args:
            text: Nội dung review (đã clean hoặc chưa)

        Returns:
            List các chủ đề tìm thấy (có thể rỗng)
        """
        if not isinstance(text, str) or not text.strip():
            return []

        text_lower = text.lower()
        found_topics = []

        for topic in self.TOPIC_ORDER:
            pattern = self.compiled_patterns[topic]
            if pattern.search(text_lower):
                found_topics.append(topic)

        return found_topics

    def categorize_primary(self, text: str) -> str:
        """
        Lấy chủ đề chính (nhiều keyword match nhất) cho 1 review.

        Args:
            text: Nội dung review

        Returns:
            Tên chủ đề chính hoặc 'Khác'
        """
        scores = self.categorize_with_scores(text)
        if not scores:
            return "Khác"
        # Return topic with highest match count; tie-break by TOPIC_ORDER
        max_score = max(scores.values())
        for topic in self.TOPIC_ORDER:
            if scores.get(topic, 0) == max_score:
                return topic
        return "Khác"

    def categorize_with_scores(self, text: str) -> dict:
        """
        Phân loại với điểm (số keyword matches) cho mỗi chủ đề.

        Args:
            text: Nội dung review

        Returns:
            Dict {topic: match_count}
        """
        if not isinstance(text, str) or not text.strip():
            return {}

        text_lower = text.lower()
        scores = {}

        for topic in self.TOPIC_ORDER:
            pattern = self.compiled_patterns[topic]
            matches = pattern.findall(text_lower)
            if matches:
                scores[topic] = len(matches)

        return scores

    def categorize_dataframe(
        self, df: pd.DataFrame, text_column: str = "content", use_cleaned: bool = True
    ) -> pd.DataFrame:
        """
        Phân loại chủ đề cho toàn bộ DataFrame.

        Args:
            df: DataFrame chứa reviews
            text_column: Tên cột chứa text gốc
            use_cleaned: Sử dụng cột content_cleaned nếu có

        Returns:
            DataFrame với các cột mới:
            - topic_primary: chủ đề chính
            - topic_all: tất cả chủ đề (dạng list)
            - topic_count: số chủ đề
        """
        df_result = df.copy()

        # Chọn cột text để phân tích
        if use_cleaned and "content_cleaned" in df_result.columns:
            col = "content_cleaned"
        elif text_column in df_result.columns:
            col = text_column
        else:
            raise ValueError(f"Không tìm thấy cột '{text_column}' trong DataFrame")

        print(f"🏷️  Đang phân loại chủ đề cho {len(df_result):,} reviews...")
        print(f"   Sử dụng cột: '{col}'")

        # Áp dụng categorization
        df_result["topic_all"] = df_result[col].apply(self.categorize_single)
        df_result["topic_primary"] = df_result["topic_all"].apply(
            lambda x: x[0] if x else "Khác"
        )
        df_result["topic_count"] = df_result["topic_all"].apply(len)

        # Thống kê
        topic_counts = df_result["topic_primary"].value_counts()
        total = len(df_result)

        print(f"\n📊 KẾT QUẢ PHÂN LOẠI CHỦ ĐỀ:")
        print(f"{'─' * 50}")
        for topic in self.TOPIC_ORDER + ["Khác"]:
            count = topic_counts.get(topic, 0)
            pct = count / total * 100
            icon = self.TOPICS.get(topic, {}).get("icon", "📌")
            print(f"   {icon} {topic:30s}: {count:6,} ({pct:5.1f}%)")

        multi_topic = (df_result["topic_count"] > 1).sum()
        no_topic = (df_result["topic_count"] == 0).sum()
        print(
            f"\n   📎 Reviews đa chủ đề: {multi_topic:,} ({multi_topic/total*100:.1f}%)"
        )
        print(f"   📭 Reviews không xác định: {no_topic:,} ({no_topic/total*100:.1f}%)")
        print(f"{'─' * 50}\n")

        return df_result

    def get_topic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo bảng thống kê chi tiết theo chủ đề.

        Args:
            df: DataFrame đã được categorize (có cột topic_primary, sentiment)

        Returns:
            DataFrame thống kê: Topic, Count, %, Positive%, Negative%, AvgRating
        """
        stats = []
        total = len(df)

        for topic in self.TOPIC_ORDER + ["Khác"]:
            topic_df = df[df["topic_primary"] == topic]
            if len(topic_df) == 0:
                continue

            count = len(topic_df)
            pct = count / total * 100

            # Sentiment breakdown
            pos_count = (topic_df["sentiment"] == "positive").sum()
            neg_count = (topic_df["sentiment"] == "negative").sum()
            pos_pct = pos_count / count * 100 if count > 0 else 0
            neg_pct = neg_count / count * 100 if count > 0 else 0

            # Average rating
            avg_rating = topic_df["score"].mean() if "score" in topic_df.columns else 0

            icon = self.TOPICS.get(topic, {}).get("icon", "📌")

            stats.append(
                {
                    "Chủ đề": topic,
                    "Icon": icon,
                    "Số reviews": count,
                    "% Tổng": round(pct, 1),
                    "% Tích cực": round(pos_pct, 1),
                    "% Tiêu cực": round(neg_pct, 1),
                    "Rating TB": round(avg_rating, 2),
                }
            )

        return pd.DataFrame(stats)

    def get_topic_bank_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo ma trận Topic × Bank (số lượng hoặc tỷ lệ).

        Returns:
            DataFrame pivot: rows=Topics, columns=Banks, values=counts
        """
        if "bank_name" not in df.columns:
            raise ValueError("DataFrame cần có cột 'bank_name'")

        matrix = pd.crosstab(df["topic_primary"], df["bank_name"], margins=False)

        # Reorder rows
        ordered_topics = [t for t in self.TOPIC_ORDER + ["Khác"] if t in matrix.index]
        matrix = matrix.reindex(ordered_topics)

        return matrix

    def get_topic_sentiment_bank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo bảng chi tiết: Topic × Bank × Sentiment ratio.
        Trả về % tích cực cho mỗi cặp (topic, bank).
        """
        if "bank_name" not in df.columns:
            raise ValueError("DataFrame cần có cột 'bank_name'")

        # Calculate positive ratio for each topic-bank pair
        result = (
            df.groupby(["topic_primary", "bank_name"])["sentiment"]
            .apply(lambda x: (x == "positive").mean() * 100)
            .unstack(fill_value=0)
            .round(1)
        )

        # Reorder
        ordered_topics = [t for t in self.TOPIC_ORDER + ["Khác"] if t in result.index]
        result = result.reindex(ordered_topics)

        return result

    def explode_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explode multi-topic reviews thành nhiều dòng (cho treemap).
        Mỗi review xuất hiện nhiều lần nếu có nhiều chủ đề.

        Returns:
            DataFrame với 1 dòng per (review, topic)
        """
        df_exploded = df.copy()

        # Replace empty lists with ['Khác']
        df_exploded["topic_all"] = df_exploded["topic_all"].apply(
            lambda x: x if x else ["Khác"]
        )

        df_exploded = df_exploded.explode("topic_all")
        df_exploded = df_exploded.rename(columns={"topic_all": "topic"})

        return df_exploded

    @classmethod
    def get_topic_color(cls, topic_name: str) -> str:
        """Lấy màu cho chủ đề."""
        if topic_name in cls.TOPICS:
            return cls.TOPICS[topic_name]["color"]
        return "#6b7280"  # Gray for 'Khác'

    @classmethod
    def get_topic_icon(cls, topic_name: str) -> str:
        """Lấy icon cho chủ đề."""
        if topic_name in cls.TOPICS:
            return cls.TOPICS[topic_name]["icon"]
        return "📌"

    @classmethod
    def get_all_colors(cls) -> dict:
        """Lấy dict màu cho tất cả chủ đề."""
        colors = {t: c["color"] for t, c in cls.TOPICS.items()}
        colors["Khác"] = "#6b7280"
        return colors


# ============================================================
# STANDALONE USAGE
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🏷️  BANKING TOPIC CATEGORIZER - TEST")
    print("=" * 60)

    categorizer = BankingTopicCategorizer()

    # Test cases
    test_reviews = [
        "App rất lag, mở mãi không vào được, đơ liên tục",
        "Chuyển tiền nhanh, giao dịch thuận tiện",
        "Đăng nhập bằng vân tay rất tiện, face id cũng ok",
        "Giao diện đẹp, dễ sử dụng, bố cục rõ ràng",
        "Tổng đài không ai nghe, gọi mãi không được, nhân viên vô trách nhiệm",
        "Sao trừ phí hoài vậy, lệ phí cao quá",
        "Quảng cáo nhiều quá, thông báo spam liên tục",
        "Cập nhật xong bị lỗi, không vào được app",
        "Bảo mật tốt, an toàn khi giao dịch",
        "Quét QR thanh toán nhanh, tính năng đa dạng",
        "tốt",
        "tệ quá",
    ]

    print("\n📝 Kết quả phân loại:")
    print("-" * 60)
    for review in test_reviews:
        topics = categorizer.categorize_single(review)
        primary = categorizer.categorize_primary(review)
        icon = categorizer.get_topic_icon(primary)
        print(f'  {icon} [{primary}] ← "{review[:60]}..."')
        if len(topics) > 1:
            print(f"      ↳ Đa chủ đề: {topics}")

    print("\n✅ Test hoàn thành!")
