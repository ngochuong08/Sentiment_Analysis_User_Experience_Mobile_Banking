# -*- coding: utf-8 -*-
"""
WEB DASHBOARD: PHÂN TÍCH CẢM XÚC NGƯỜI DÙNG ĐỐI VỚI CÁC ỨNG DỤNG NGÂN HÀNG DI ĐỘNG TẠI VIỆT NAM
Based Sentiment Analysis of User Experience in Mobile Banking Applications on App Store
-
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from underthesea import word_tokenize
import json
from datetime import datetime
import re

# Topic categorization
from topic_categorizer import BankingTopicCategorizer

# ========================================
# CẤU HÌNH TRANG
# ========================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tùy chỉnh
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        padding: 20px 0;
    }
    h2 {
        color: #1e40af;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 10px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ========================================
# HÀM LOAD DỮ LIỆU
# ========================================
@st.cache_data
def load_data():
    """Load dữ liệu reviews"""
    try:
        df = pd.read_csv("./output/reviews_with_sentiment.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file reviews_with_sentiment.csv")
        return None


@st.cache_data
def load_bank_stats():
    """Load thống kê theo ngân hàng"""
    try:
        df = pd.read_csv("./output/bank_statistics.csv", index_col=0)
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def load_model_comparison():
    """Load kết quả so sánh mô hình"""
    try:
        df = pd.read_csv("./output/model_comparison_advanced.csv")
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_model():
    """Load mô hình ML và vectorizers (word + char)"""
    try:
        # Load improved model
        with open("./output/best_model_advanced.pkl", "rb") as f:
            model = pickle.load(f)
        # Load word-level TF-IDF
        with open("./output/tfidf_word_vectorizer.pkl", "rb") as f:
            tfidf_word = pickle.load(f)
        # Load char-level TF-IDF
        with open("./output/tfidf_char_vectorizer.pkl", "rb") as f:
            tfidf_char = pickle.load(f)
        # Load metadata
        with open("./output/model_metadata_advanced.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, tfidf_word, tfidf_char, metadata
    except FileNotFoundError as e:
        st.error(f"⚠️ Không tìm thấy file model: {e}")
        return None, None, None, None


# ========================================
# CLASS ADVANCED VIETNAMESE REVIEW CLEANER
# ========================================
import unicodedata


# ========================================
# 3. ADVANCED VIETNAMESE TEXT CLEANER
# ========================================
class AdvancedVietnameseReviewCleaner:
    """
    Bộ xử lý text nâng cao cho tiếng Việt
    - 150+ teencode words
    - Unicode normalization (NFC)
    - Vietnamese stopwords removal
    - Advanced typo correction
    """

    def __init__(self):
        # Extended teencode dictionary (150+ words)
        self.teencode_dict = {
            # Phủ định
            "k": "không",
            "ko": "không",
            "hok": "không",
            "hong": "không",
            "hem": "không",
            "kg": "không",
            "kh": "không",
            "khong": "không",
            "hông": "không",
            "kô": "không",
            "hỏng": "không",
            "khôg": "không",
            "chả": "chẳng",
            "chăng": "chẳng",
            "chẵng": "chẳng",
            # Được
            "dc": "được",
            "đc": "được",
            "dk": "được",
            "đk": "được",
            "duoc": "được",
            "đươc": "được",
            "dược": "được",
            # Với, vậy, vì
            "vs": "với",
            "vc": "với",
            "v": "với",
            "z": "vậy",
            "zay": "vậy",
            "zị": "vậy",
            "vk": "vì",
            "vic": "vì",
            # Mình, tôi
            "ms": "mới",
            "mik": "mình",
            "mk": "mình",
            "mh": "mình",
            "mjh": "mình",
            "tui": "tôi",
            "toy": "tôi",
            "toj": "tôi",
            "tớ": "tôi",
            "t": "tôi",
            "mik": "mình",
            # Bình thường, như thế nào
            "bt": "bình thường",
            "bth": "bình thường",
            "ntn": "như thế nào",
            "sao": "như thế nào",
            "nthna": "như thế nào",
            "nhma": "nhưng mà",
            "nma": "nhưng mà",
            # Rồi, nhé
            "r": "rồi",
            "rùi": "rồi",
            "rui": "rồi",
            "ròi": "rồi",
            "ak": "à",
            "ạk": "ạ",
            "nhaa": "nhé",
            "nha": "nhé",
            "nek": "nè",
            "né": "nè",
            "òi": "rồi",
            "oy": "rồi",
            # Cũng, biết
            "cx": "cũng",
            "cug": "cũng",
            "bik": "biết",
            "bit": "biết",
            "bjt": "biết",
            "biet": "biết",
            # Quá, gì
            "wa": "quá",
            "qá": "quá",
            "wá": "quá",
            "qu": "quá",
            "j": "gì",
            "zì": "gì",
            "jì": "gì",
            "dzì": "gì",
            "ji": "gì",
            "gi": "gì",
            # Chưa
            "chs": "chưa",
            "chx": "chưa",
            "chwa": "chưa",
            # Cảm ơn
            "tks": "cảm ơn",
            "tk": "cảm ơn",
            "thanks": "cảm ơn",
            "tnks": "cảm ơn",
            "cam on": "cảm ơn",
            "tks": "cảm ơn",
            # Nói chuyện
            "nc": "nói chuyện",
            "nch": "nói chuyện",
            # Người ta, người
            "ngta": "người ta",
            "nguoi ta": "người ta",
            "nta": "người ta",
            "ng": "người",
            "nguoi": "người",
            # Ừ, ok
            "uk": "ừ",
            "uh": "ừ",
            "ừm": "ừ",
            "oke": "ok",
            "okie": "ok",
            "okee": "ok",
            "okey": "ok",
            "okela": "ok",
            # Đang, làm
            "đag": "đang",
            "dg": "đang",
            "dang": "đang",
            "lm": "làm",
            "lam": "làm",
            "lam": "làm",
            # Phải
            "pk": "phải",
            "fai": "phải",
            "pải": "phải",
            "fải": "phải",
            # Xin lỗi
            "xl": "xin lỗi",
            "sr": "xin lỗi",
            "sorry": "xin lỗi",
            "sry": "xin lỗi",
            # Trả lời
            "tl": "trả lời",
            "rep": "trả lời",
            "reply": "trả lời",
            # Inbox, nhắn tin
            "ib": "inbox",
            "mess": "nhắn tin",
            "msg": "nhắn tin",
            # Sử dụng
            "sd": "sử dụng",
            "xd": "sử dụng",
            # Bạn
            "bn": "bạn",
            "b": "bạn",
            "bợn": "bạn",
            # Mọi người, admin
            "mn": "mọi người",
            "ad": "admin",
            "adm": "admin",
            # App, ứng dụng
            "app": "ứng dụng",
            "ứg dụng": "ứng dụng",
            "ung dung": "ứng dụng",
            # Nhiều, một
            "nhìu": "nhiều",
            "nhiu": "nhiều",
            "nhìu": "nhiều",
            "1": "một",
            "mote": "một",
            "mốt": "một",
            # Vào, ra
            "zô": "vào",
            "zo": "vào",
            "zào": "vào",
            "wào": "vào",
            # Trước, sau
            "trc": "trước",
            "tr": "trước",
            "trướck": "trước",
            "sau": "sau",
            "sao": "sau",
            # Luôn
            "lun": "luôn",
            "luôn": "luôn",
            # Nữa
            "nx": "nữa",
            "nax": "nữa",
            "nưa": "nữa",
            # Thích, muốn
            "thik": "thích",
            "thix": "thích",
            "muon": "muốn",
            "mún": "muốn",
            # Cái, của
            "cai": "cái",
            "cá": "cái",
            "cua": "của",
            "của": "của",
            # Hay, tốt, xấu
            "hay": "hay",
            "haii": "hay",
            "tot": "tốt",
            "tốtt": "tốt",
            "xau": "xấu",
            "te": "tệ",
        }

        # Extended typo dictionary
        self.typo_dict = {
            "nhu": "như",
            "chi": "chỉ",
            "nhung": "nhưng",
            "ma": "mà",
            "thi": "thì",
            "tai": "tại",
            "lai": "lại",
            "nua": "nữa",
            "dau": "đâu",
            "khi": "khi",
            "rat": "rất",
            "den": "đến",
            "cho": "cho",
            "chua": "chưa",
            "roi": "rồi",
            "vao": "vào",
            "xem": "xem",
            "nhan": "nhận",
            "gui": "gửi",
            "giap diện": "giao diện",
            "giap dien": "giao diện",
            "gianh diện": "giao diện",
        }

        # Vietnamese stopwords
        self.stopwords = set(
            [
                "và",
                "của",
                "có",
                "thì",
                "là",
                "được",
                "hoặc",
                "các",
                "này",
                "đó",
                "những",
                "cho",
                "từ",
                "trong",
                "nếu",
                "khi",
                "mà",
                "đã",
                "sẽ",
                "để",
                "với",
                "bởi",
                "về",
                "như",
                "tại",
                "hay",
                "nhưng",
                "đến",
                "còn",
                "thế",
                "nào",
                "ai",
                "gì",
            ]
        )

        # Negation words (phủ định)
        self.negation_words = {
            "không",
            "chưa",
            "chẳng",
            "chả",
            "không bao giờ",
            "chưa bao giờ",
            "đừng",
            "đừng có",
            "không phải",
            "chẳng phải",
            "không hề",
            "chẳng hề",
            "không có",
            "chả có",
            "không còn",
            "không thể",
            "chưa thể",
            "không nên",
            "chưa nên",
            "không được",
            "chưa được",
            "không ai",
            "chẳng ai",
            "không gì",
            "chẳng gì",
            "k thể",
            "ko thể",
            "k đc",
            "ko đc",
            "k thể",
            "ko thể",
        }

        # Negative words that become positive with negation
        # "không tệ" = tốt, "chưa tốt" = tệ
        self.negative_words = {
            "tệ",
            "xấu",
            "dở",
            "kém",
            "tồi",
            "tệ hại",
            "xấu xí",
            "tệ quá",
            "dở quá",
            "kém quá",
            "tồi tệ",
            "tệ nhất",
            "xấu nhất",
            "dở nhất",
            "kém nhất",
            "thất vọng",
            "tệ lắm",
            "kém cỏi",
            "dở ẹc",
            "rác",
            "rác rưởi",
            "thất bại",
            "tệ hại",
            "ngớ ngẩn",
            "ngu",
            "lỗi",
            "lỗi nhiều",
            "lag",
            "giật lag",
            "đơ",
            "treo",
            "lắc",
            "văng",
            "crash",
            "lỗi thường xuyên",
            "chậm",
            "chậm chạp",
            "cùi",
            "cùi bắp",
            "quá tệ",
            "tệ quá đi",
            "không ổn",
            "không tốt",
            "ko",
            "k",
            "k thể nào",
            "ko thể",
            "không thể",
            "xàm",
            "xàm xí",
            "rối",
            "bực",
            "ức chế",
            "phiền phức",
            "rắc rối",
            "mất thời gian",
            "khó chịu",
            "đáng ghét",
            "khó dùng",
            "khó sử dụng",
            "ghét",
            "dở dở ương ương",
            "đơ đơ",
            "lag lag",
            "giật giật",
            "chập chờn",
            "điên",
            "phí phạm",
            "phí thời gian",
            "lằng nhằng",
            "rối rắm",
            "lộn xộn",
            "hỏng",
            "đóng băng",
            "đơ máy",
            "đơ ứng dụng",
            "đơ app",
            "đứng hình",
            "bị đơ",
            "không load được",
            "không đăng nhập được",
            "không mở được",
            "không sử dụng được",
            "không vào được",
            "mất kết nối",
            "mất mạng",
            "mất tín hiệu",
            "sập nguồn",
            "sập máy",
            "sập app",
            "sập ứng dụng",
            "treo máy",
            "treo app",
            "treo ứng dụng",
            "chậm kinh khủng",
            "chậm kinh",
            "chậm vãi",
            "bất tiện",
            "rối mắt",
        }

        # Positive words that become negative with negation
        # "không tốt" = tệ, "chưa hay" = dở
        self.positive_words = {
            "tốt",
            "hay",
            "đẹp",
            "ổn",
            "ok",
            "oke",
            "okela",
            "mượt",
            "nhanh",
            "tiện",
            "tiện lợi",
            "tốt lắm",
            "hay lắm",
            "tuyệt",
            "tuyệt vời",
            "xuất sắc",
            "hoàn hảo",
            "ưng",
            "ưng ý",
            "hài lòng",
            "tốt quá",
            "hay quá",
            "đẹp quá",
            "mượt mà",
            "nhanh chóng",
            "tiện ích",
            "ổn định",
            "bền",
            "chất lượng",
            "tuyệt hảo",
            "tốt nhất",
            "hay nhất",
            "đẹp nhất",
            "mượt nhất",
            "nhanh nhất",
            "tiện nhất",
            "ưng nhất",
            "xuất sắc nhất",
            "best",
            "perfect",
            "excellent",
            "amazing",
            "fantastic",
            "awesome",
            "great",
            "love",
            "loved",
            "loving",
            "like",
            "liked",
            "liking",
            "được",
            "đc",
            "dc",
        }

        # Emoji pattern
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

    def normalize_unicode(self, text: str) -> str:
        """Unicode normalization (NFC)"""
        return unicodedata.normalize("NFC", text)

    def handle_negation(self, text: str) -> str:
        """
        Xử lý phủ định trong tiếng Việt
        VD: "không tệ" -> "POSNEG_tệ" (tích cực)
            "không tốt" -> "NEGNEG_tốt" (tiêu cực)
        """
        words = text.split()
        result = []
        i = 0

        while i < len(words):
            current_word = words[i]

            # Check if current word is negation
            if current_word in self.negation_words:
                # Look ahead for next 1-3 words
                negation_phrase = current_word
                found_sentiment = False

                # Check next 3 words for sentiment words
                for j in range(i + 1, min(i + 4, len(words))):
                    next_word = words[j]
                    negation_phrase += " " + next_word

                    # Check if it's a negative word (không + negative = positive)
                    if next_word in self.negative_words:
                        # Transform: "không tệ" -> "POSNEG_tệ" (positive negation)
                        result.append(f"POSNEG_{next_word}")
                        i = j + 1
                        found_sentiment = True
                        break

                    # Check if it's a positive word (không + positive = negative)
                    elif next_word in self.positive_words:
                        # Transform: "không tốt" -> "NEGNEG_tốt" (negative negation)
                        result.append(f"NEGNEG_{next_word}")
                        i = j + 1
                        found_sentiment = True
                        break

                # If no sentiment word found, keep original negation
                if not found_sentiment:
                    result.append(current_word)
                    i += 1
            else:
                result.append(current_word)
                i += 1

        return " ".join(result)

    def clean_text(self, text: str, remove_stopwords: bool = False) -> str:
        """Làm sạch text với nhiều bước xử lý"""
        if not isinstance(text, str) or not text.strip():
            return ""

        # Unicode normalization
        text = self.normalize_unicode(text)

        # Lowercase
        text = text.lower()

        # Remove emoji
        text = self.emoji_pattern.sub("", text)

        # Remove special characters (keep Vietnamese)
        text = re.sub(
            r"[^\w\s\.,!?áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]",
            " ",
            text,
        )

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove repeated characters (3+ times)
        text = re.sub(r"(.)\1{2,}", r"\1", text)

        # Apply teencode dictionary
        words = text.split()
        words = [self.teencode_dict.get(w, w) for w in words]
        text = " ".join(words)

        # Apply typo dictionary
        for wrong, correct in self.typo_dict.items():
            text = text.replace(wrong, correct)

        # 🔥 HANDLE NEGATION (before stopwords removal)
        text = self.handle_negation(text)

        # Remove stopwords if requested
        if remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = " ".join(words)

        return " ".join(text.split())

    def clean_dataframe(
        self, df: pd.DataFrame, text_column: str = "content"
    ) -> pd.DataFrame:
        """Làm sạch toàn bộ DataFrame"""
        df_cleaned = df.copy()
        print(f"🧹 Đang làm sạch {len(df_cleaned):,} reviews...")

        df_cleaned[f"{text_column}_cleaned"] = df_cleaned[text_column].apply(
            lambda x: self.clean_text(x, remove_stopwords=False)
        )

        empty_reviews = df_cleaned[f"{text_column}_cleaned"].str.strip().eq("").sum()
        print(f"✅ Hoàn thành!")
        print(f"   • Reviews trống: {empty_reviews:,}")
        print(f"   • Reviews hợp lệ: {len(df_cleaned) - empty_reviews:,}\n")

        return df_cleaned


# ========================================
# MAPPING TÊN NGÂN HÀNG
# ========================================
APP_NAMES = {
    "com.vnpay.bidv": "BIDV",
    "com.vnpay.Agribank3g": "Agribank",
    "com.VCB": "Vietcombank",
    "com.sacombank.ewallet": "Sacombank",
    "com.mbmobile": "MBBank",
    "com.vietinbank.ipay": "VietinBank",
    "com.tpb.mb.gprsandroid": "TPBank",
    "vn.com.techcombank.bb.app": "Techcombank",
    "com.vnpay.hdbank": "HDBank",
    "mobile.acb.com.vn": "ACB",
    "vn.com.ocb.awe": "OCB",
    "vn.com.msb.smartBanking": "MSB",
}

# Khởi tạo Advanced text cleaner
TEXT_CLEANER = AdvancedVietnameseReviewCleaner()


# ========================================
# HÀM DỰ ĐOÁN SENTIMENT
# ========================================
def predict_sentiment(text, model, tfidf_word, tfidf_char):
    """Dự đoán sentiment cho text input (IMPROVED with word + char TF-IDF)"""
    from scipy.sparse import hstack

    if not text.strip():
        return None, None, None, None

    # Bước 1: Làm sạch text (giống như lúc training)
    text_cleaned = TEXT_CLEANER.clean_text(text)

    # Bước 2: Tách từ tiếng Việt
    text_segmented = word_tokenize(text_cleaned, format="text")

    # Bước 3: Vectorize với cả word và char TF-IDF
    text_word = tfidf_word.transform([text_segmented])
    text_char = tfidf_char.transform([text_segmented])
    text_vectorized = hstack([text_word, text_char])

    # Bước 4: Dự đoán
    prediction = model.predict(text_vectorized)[0]

    # Bước 5: Lấy probability (nếu model hỗ trợ)
    has_real_confidence = False
    try:
        proba = model.predict_proba(text_vectorized)[0]
        confidence = max(proba) * 100
        has_real_confidence = True
    except Exception as e:
        # Fallback: Use default confidence
        # This happens when model doesn't support predict_proba (e.g., LinearSVC, hard voting)
        confidence = 75.0  # Default moderate confidence
        has_real_confidence = False

    sentiment = "Tích cực 😊" if prediction == 1 else "Tiêu cực 😞"

    # Return debug info + confidence type flag
    return sentiment, confidence, text_cleaned, text_segmented, has_real_confidence
    return sentiment, confidence, text_cleaned, text_segmented


# ========================================
# HEADER
# ========================================
st.title("📊 Sentiment Analysis Dashboard for Mobile Banking Applications")
st.markdown("---")

# ========================================
# SIDEBAR
# ========================================
st.sidebar.title("⚙️ Cài đặt")
st.sidebar.markdown("---")

# Load dữ liệu
df = load_data()
bank_stats = load_bank_stats()
model_comparison = load_model_comparison()
model, tfidf_word, tfidf_char, metadata = load_model()

if df is not None:
    # Thêm cột tên ngân hàng
    if "bank_name" not in df.columns and "appId" in df.columns:
        df["bank_name"] = df["appId"].map(APP_NAMES)

    # Sidebar filters
    st.sidebar.subheader("🔍 Bộ lọc dữ liệu")

    # Filter theo ngân hàng
    banks = ["Tất cả"] + sorted(df["bank_name"].dropna().unique().tolist())
    selected_bank = st.sidebar.selectbox("Chọn ngân hàng:", banks)

    # Filter theo sentiment
    sentiments = ["Tất cả", "Tích cực", "Tiêu cực"]
    selected_sentiment = st.sidebar.selectbox("Chọn cảm xúc:", sentiments)

    # Filter theo rating
    min_rating, max_rating = st.sidebar.slider(
        "Chọn khoảng rating:", min_value=1, max_value=5, value=(1, 5)
    )

    # Áp dụng filter
    df_filtered = df.copy()

    if selected_bank != "Tất cả":
        df_filtered = df_filtered[df_filtered["bank_name"] == selected_bank]

    if selected_sentiment == "Tích cực":
        df_filtered = df_filtered[df_filtered["sentiment"] == "positive"]
    elif selected_sentiment == "Tiêu cực":
        df_filtered = df_filtered[df_filtered["sentiment"] == "negative"]

    df_filtered = df_filtered[
        (df_filtered["score"] >= min_rating) & (df_filtered["score"] <= max_rating)
    ]

    st.sidebar.markdown("---")
    st.sidebar.info(f"📝 Hiển thị {len(df_filtered):,} / {len(df):,} reviews")

    # ========================================
    # TAB NAVIGATION
    # ========================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📊 Tổng quan",
            "🏦 Phân tích theo ngân hàng",
            "🏷️ Phân tích chủ đề",
            "💹 Sentiment × Cổ phiếu",
            "🤖 So sánh mô hình ML",
            "🔮 Dự đoán Sentiment",
            "📝 Dữ liệu chi tiết",
        ]
    )

    # ========================================
    # TAB 1: TỔNG QUAN
    # ========================================
    with tab1:
        st.header("📊 Tổng quan dữ liệu")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        total_reviews = len(df_filtered)
        positive_reviews = len(df_filtered[df_filtered["sentiment"] == "positive"])
        negative_reviews = len(df_filtered[df_filtered["sentiment"] == "negative"])
        avg_rating = df_filtered["score"].mean()

        with col1:
            st.metric("📝 Tổng reviews", f"{total_reviews:,}")
        with col2:
            st.metric(
                "😊 Reviews tích cực",
                f"{positive_reviews:,}",
                delta=f"{positive_reviews/total_reviews*100:.1f}%",
            )
        with col3:
            st.metric(
                "😞 Reviews tiêu cực",
                f"{negative_reviews:,}",
                delta=f"{negative_reviews/total_reviews*100:.1f}%",
                delta_color="inverse",
            )
        with col4:
            st.metric("⭐ Rating trung bình", f"{avg_rating:.2f}")

        st.markdown("---")

        # Biểu đồ
        col1, col2 = st.columns(2)

        with col1:
            # Phân bố sentiment
            sentiment_counts = df_filtered["sentiment"].value_counts()
            sentiment_labels = [
                "Tích cực" if s == "positive" else "Tiêu cực"
                for s in sentiment_counts.index
            ]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=sentiment_labels,
                        values=sentiment_counts.values,
                        hole=0.4,
                        marker=dict(colors=["#10b981", "#ef4444"]),
                        textinfo="label+percent",
                        textfont=dict(size=14, color="white", family="Arial Black"),
                    )
                ]
            )

            fig.update_layout(
                title="<b>Phân bố Sentiment</b>",
                title_font=dict(size=18, color="#1e40af"),
                height=400,
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Phân bố rating
            rating_counts = df_filtered["score"].value_counts().sort_index()

            colors = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#10b981"]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=rating_counts.index,
                        y=rating_counts.values,
                        marker=dict(color=colors, line=dict(color="black", width=2)),
                        text=rating_counts.values,
                        textposition="outside",
                        textfont=dict(size=12, color="black", family="Arial Black"),
                    )
                ]
            )

            fig.update_layout(
                title="<b>Phân bố Rating (1-5 sao)</b>",
                title_font=dict(size=18, color="#1e40af"),
                xaxis_title="Rating (sao)",
                yaxis_title="Số lượng reviews",
                height=400,
                xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            )

            st.plotly_chart(fig, use_container_width=True)

        # Xu hướng theo thời gian
        if "at" in df_filtered.columns:
            st.markdown("### 📈 Xu hướng Sentiment theo thời gian")

            df_time = df_filtered.copy()
            df_time["at"] = pd.to_datetime(df_time["at"])
            df_time["month"] = df_time["at"].dt.to_period("M").astype(str)

            time_sentiment = (
                df_time.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=time_sentiment.index,
                    y=time_sentiment.get("positive", 0),
                    name="Tích cực",
                    mode="lines+markers",
                    line=dict(color="#10b981", width=3),
                    marker=dict(size=8),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=time_sentiment.index,
                    y=time_sentiment.get("negative", 0),
                    name="Tiêu cực",
                    mode="lines+markers",
                    line=dict(color="#ef4444", width=3),
                    marker=dict(size=8),
                )
            )

            fig.update_layout(
                title="<b>Xu hướng số lượng reviews theo tháng</b>",
                title_font=dict(size=18, color="#1e40af"),
                xaxis_title="Tháng",
                yaxis_title="Số lượng reviews",
                height=400,
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

    # ========================================
    # TAB 2: PHÂN TÍCH THEO NGÂN HÀNG
    # ========================================
    with tab2:
        st.header("🏦 Phân tích theo từng ngân hàng")

        if bank_stats is not None:
            # Sắp xếp theo rating
            bank_stats = bank_stats.sort_values("Rating TB", ascending=False)

            # Biểu đồ so sánh rating
            st.markdown("### ⭐ Rating trung bình các ngân hàng")

            fig = go.Figure()

            colors = [
                "#10b981" if r >= 4 else "#eab308" if r >= 3 else "#ef4444"
                for r in bank_stats["Rating TB"]
            ]

            fig.add_trace(
                go.Bar(
                    y=bank_stats.index.tolist(),
                    x=bank_stats["Rating TB"],
                    orientation="h",
                    marker=dict(color=colors, line=dict(color="black", width=2)),
                    text=[f"{r:.2f}⭐" for r in bank_stats["Rating TB"]],
                    textposition="outside",
                    textfont=dict(size=12, color="black", family="Arial Black"),
                )
            )

            fig.update_layout(
                title="<b>Rating trung bình theo ngân hàng</b>",
                title_font=dict(size=18, color="#1e40af"),
                xaxis_title="Rating",
                yaxis_title="Ngân hàng",
                height=500,
                xaxis=dict(range=[0, 5.5]),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Biểu đồ tỷ lệ tích cực
            st.markdown("### 😊 Tỷ lệ reviews tích cực")

            fig = go.Figure()

            colors = [
                "#10b981" if p >= 50 else "#ef4444" for p in bank_stats["% Tích cực"]
            ]

            fig.add_trace(
                go.Bar(
                    y=bank_stats.index.tolist(),
                    x=bank_stats["% Tích cực"],
                    orientation="h",
                    marker=dict(color=colors, line=dict(color="black", width=2)),
                    text=[f"{p:.1f}%" for p in bank_stats["% Tích cực"]],
                    textposition="outside",
                    textfont=dict(size=12, color="black", family="Arial Black"),
                )
            )

            fig.update_layout(
                title="<b>Tỷ lệ % reviews tích cực</b>",
                title_font=dict(size=18, color="#1e40af"),
                xaxis_title="% Tích cực",
                yaxis_title="Ngân hàng",
                height=500,
                xaxis=dict(range=[0, 110]),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Bảng thống kê chi tiết
            st.markdown("### 📋 Bảng thống kê chi tiết")
            st.dataframe(
                bank_stats.style.format(
                    {
                        "Số reviews": "{:,.0f}",
                        "Rating TB": "{:.2f}",
                        "% Tích cực": "{:.1f}%",
                    }
                )
                .background_gradient(subset=["Rating TB"], cmap="RdYlGn")
                .background_gradient(subset=["% Tích cực"], cmap="RdYlGn"),
                use_container_width=True,
            )
        else:
            st.warning("⚠️ Chưa có dữ liệu thống kê theo ngân hàng")

    # ========================================
    # TAB 3: PHÂN TÍCH CHỦ ĐỀ (TOPIC ANALYSIS)
    # ========================================
    with tab3:
        st.header("🏷️ Phân tích chủ đề reviews")
        st.markdown(
            """
        > **Phân loại chủ đề** giúp hiểu người dùng đang nói gì về app ngân hàng —
        > không chỉ cảm xúc tích cực/tiêu cực mà còn **nội dung cụ thể** (giao dịch, đăng nhập, phí, lỗi kỹ thuật...).
        """
        )

        # --- Initialize topic categorizer ---
        topic_categorizer = BankingTopicCategorizer()

        # Apply topic categorization on the filtered data
        df_topics = df_filtered.copy()
        text_col = (
            "content_cleaned" if "content_cleaned" in df_topics.columns else "content"
        )
        df_topics["topic_all"] = df_topics[text_col].apply(
            topic_categorizer.categorize_single
        )
        df_topics["topic_primary"] = df_topics["topic_all"].apply(
            lambda x: x[0] if x else "Khác"
        )
        # Use score-based primary for better accuracy
        df_topics["topic_primary"] = df_topics[text_col].apply(
            topic_categorizer.categorize_primary
        )
        df_topics["topic_count"] = df_topics["topic_all"].apply(len)

        # --- METRICS ROW ---
        st.markdown("### 📊 Tổng quan chủ đề")
        total_with_topic = (df_topics["topic_primary"] != "Khác").sum()
        multi_topic = (df_topics["topic_count"] > 1).sum()
        unique_topics = df_topics["topic_primary"].nunique()

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("📝 Tổng reviews", f"{len(df_topics):,}")
        with mcol2:
            st.metric(
                "🏷️ Có chủ đề",
                f"{total_with_topic:,}",
                delta=f"{total_with_topic/len(df_topics)*100:.1f}%",
            )
        with mcol3:
            st.metric(
                "📎 Đa chủ đề",
                f"{multi_topic:,}",
                delta=f"{multi_topic/len(df_topics)*100:.1f}%",
            )
        with mcol4:
            st.metric("🔢 Số chủ đề", f"{unique_topics}")

        st.markdown("---")

        # ============================================
        # 3.1: TREEMAP - Chủ đề × Ngân hàng
        # ============================================
        st.markdown("### 🗺️ Treemap: Phân bố chủ đề theo ngân hàng")
        st.caption(
            "Kích thước ô = số lượng reviews. Màu sắc = tỷ lệ tích cực (xanh=tích cực, đỏ=tiêu cực)"
        )

        # Prepare treemap data: explode multi-topics
        df_treemap_base = df_topics.copy()
        df_treemap_base["topic_for_treemap"] = df_treemap_base["topic_all"].apply(
            lambda x: x if x else ["Khác"]
        )
        df_treemap = df_treemap_base.explode("topic_for_treemap")

        # Aggregate: count and positive ratio per (topic, bank)
        treemap_agg = (
            df_treemap.groupby(["topic_for_treemap", "bank_name"])
            .agg(
                count=("sentiment", "size"),
                positive_ratio=("sentiment", lambda x: (x == "positive").mean() * 100),
            )
            .reset_index()
        )

        treemap_agg.columns = ["Chủ đề", "Ngân hàng", "Số reviews", "% Tích cực"]

        fig_treemap = px.treemap(
            treemap_agg,
            path=["Chủ đề", "Ngân hàng"],
            values="Số reviews",
            color="% Tích cực",
            color_continuous_scale=["#ef4444", "#fbbf24", "#10b981"],
            color_continuous_midpoint=50,
            title="<b>Treemap: Chủ đề × Ngân hàng</b>",
        )
        fig_treemap.update_layout(
            height=650,
            title_font=dict(size=18, color="#1e40af"),
            coloraxis_colorbar=dict(title="% Tích cực"),
        )
        fig_treemap.update_traces(
            textinfo="label+value",
            textfont=dict(size=13),
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

        st.markdown("---")

        # ============================================
        # 3.2: BAR CHART - Phân bố chủ đề
        # ============================================
        st.markdown("### 📊 Phân bố số lượng reviews theo chủ đề")

        col_bar1, col_bar2 = st.columns([3, 2])

        with col_bar1:
            topic_counts = df_topics["topic_primary"].value_counts()

            # Get colors and icons
            topic_colors = []
            topic_labels = []
            for t in topic_counts.index:
                topic_colors.append(BankingTopicCategorizer.get_topic_color(t))
                icon = BankingTopicCategorizer.get_topic_icon(t)
                topic_labels.append(f"{icon} {t}")

            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        y=topic_labels[::-1],
                        x=topic_counts.values[::-1],
                        orientation="h",
                        marker=dict(
                            color=topic_colors[::-1],
                            line=dict(color="black", width=1.5),
                        ),
                        text=[
                            f"{v:,} ({v/len(df_topics)*100:.1f}%)"
                            for v in topic_counts.values[::-1]
                        ],
                        textposition="outside",
                        textfont=dict(size=11, family="Arial"),
                    )
                ]
            )

            fig_bar.update_layout(
                title="<b>Số lượng reviews theo chủ đề</b>",
                title_font=dict(size=16, color="#1e40af"),
                xaxis_title="Số lượng reviews",
                height=500,
                margin=dict(l=250),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_bar2:
            # Donut chart cho top chủ đề
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=topic_counts.index.tolist(),
                        values=topic_counts.values,
                        hole=0.45,
                        marker=dict(colors=topic_colors),
                        textinfo="percent",
                        textfont=dict(size=12, color="white", family="Arial Black"),
                        insidetextorientation="radial",
                    )
                ]
            )
            fig_donut.update_layout(
                title="<b>Tỷ lệ chủ đề</b>",
                title_font=dict(size=16, color="#1e40af"),
                height=500,
                showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")

        # ============================================
        # 3.3: STACKED BAR - Sentiment per Topic
        # ============================================
        st.markdown("### 😊😞 Sentiment theo từng chủ đề")

        topic_sentiment = (
            df_topics.groupby("topic_primary")["sentiment"]
            .value_counts()
            .unstack(fill_value=0)
        )
        if "positive" not in topic_sentiment.columns:
            topic_sentiment["positive"] = 0
        if "negative" not in topic_sentiment.columns:
            topic_sentiment["negative"] = 0

        topic_sentiment["total"] = (
            topic_sentiment["positive"] + topic_sentiment["negative"]
        )
        topic_sentiment["pos_pct"] = (
            topic_sentiment["positive"] / topic_sentiment["total"] * 100
        ).round(1)
        topic_sentiment["neg_pct"] = (
            topic_sentiment["negative"] / topic_sentiment["total"] * 100
        ).round(1)
        topic_sentiment = topic_sentiment.sort_values("total", ascending=True)

        fig_stack = go.Figure()

        fig_stack.add_trace(
            go.Bar(
                y=topic_sentiment.index,
                x=topic_sentiment["pos_pct"],
                name="Tích cực",
                orientation="h",
                marker=dict(color="#10b981"),
                text=[f"{v:.1f}%" for v in topic_sentiment["pos_pct"]],
                textposition="inside",
                textfont=dict(size=11, color="white", family="Arial Black"),
            )
        )

        fig_stack.add_trace(
            go.Bar(
                y=topic_sentiment.index,
                x=topic_sentiment["neg_pct"],
                name="Tiêu cực",
                orientation="h",
                marker=dict(color="#ef4444"),
                text=[f"{v:.1f}%" for v in topic_sentiment["neg_pct"]],
                textposition="inside",
                textfont=dict(size=11, color="white", family="Arial Black"),
            )
        )

        fig_stack.update_layout(
            title="<b>Tỷ lệ tích cực/tiêu cực theo chủ đề</b>",
            title_font=dict(size=16, color="#1e40af"),
            xaxis_title="Tỷ lệ %",
            height=500,
            barmode="stack",
            xaxis=dict(range=[0, 105]),
            margin=dict(l=250),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        st.markdown("---")

        # ============================================
        # 3.4: HEATMAP - Chủ đề × Ngân hàng
        # ============================================
        st.markdown("### 🔥 Heatmap: Chủ đề × Ngân hàng (% Tích cực)")
        st.caption("Màu xanh = tỷ lệ tích cực cao. Màu đỏ = tỷ lệ tiêu cực cao.")

        # Build heatmap data
        heatmap_data = (
            df_topics.groupby(["topic_primary", "bank_name"])
            .apply(lambda x: (x["sentiment"] == "positive").mean() * 100)
            .unstack(fill_value=50)
        )

        # Reorder
        ordered_topics = [
            t
            for t in topic_categorizer.TOPIC_ORDER + ["Khác"]
            if t in heatmap_data.index
        ]
        heatmap_data = heatmap_data.reindex(ordered_topics)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale=[[0, "#ef4444"], [0.5, "#fef3c7"], [1, "#10b981"]],
                zmid=50,
                text=heatmap_data.values.round(1),
                texttemplate="%{text:.1f}%",
                textfont=dict(size=11, color="black"),
                colorbar=dict(title="% Tích cực"),
            )
        )

        fig_heatmap.update_layout(
            title="<b>Tỷ lệ tích cực theo Chủ đề × Ngân hàng (%)</b>",
            title_font=dict(size=16, color="#1e40af"),
            xaxis_title="Ngân hàng",
            yaxis_title="Chủ đề",
            height=550,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=250),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")

        # ============================================
        # 3.5: HEATMAP - Số lượng reviews
        # ============================================
        st.markdown("### 📈 Heatmap: Số lượng reviews theo Chủ đề × Ngân hàng")

        count_matrix = pd.crosstab(df_topics["topic_primary"], df_topics["bank_name"])
        count_matrix = count_matrix.reindex(
            [
                t
                for t in topic_categorizer.TOPIC_ORDER + ["Khác"]
                if t in count_matrix.index
            ]
        )

        fig_count_heatmap = go.Figure(
            data=go.Heatmap(
                z=count_matrix.values,
                x=count_matrix.columns.tolist(),
                y=count_matrix.index.tolist(),
                colorscale="Blues",
                text=count_matrix.values,
                texttemplate="%{text}",
                textfont=dict(size=10, color="black"),
                colorbar=dict(title="Số reviews"),
            )
        )

        fig_count_heatmap.update_layout(
            title="<b>Số lượng reviews theo Chủ đề × Ngân hàng</b>",
            title_font=dict(size=16, color="#1e40af"),
            xaxis_title="Ngân hàng",
            yaxis_title="Chủ đề",
            height=550,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=250),
        )
        st.plotly_chart(fig_count_heatmap, use_container_width=True)

        st.markdown("---")

        # ============================================
        # 3.6: TOP ISSUES PER BANK
        # ============================================
        st.markdown("### 🏦 Chủ đề nổi bật theo từng ngân hàng")

        selected_bank_topic = st.selectbox(
            "Chọn ngân hàng để xem chi tiết chủ đề:",
            sorted(df_topics["bank_name"].dropna().unique().tolist()),
            key="topic_bank_select",
        )

        bank_topic_df = df_topics[df_topics["bank_name"] == selected_bank_topic]

        col_bt1, col_bt2 = st.columns(2)

        with col_bt1:
            bt_counts = bank_topic_df["topic_primary"].value_counts()
            bt_colors = [
                BankingTopicCategorizer.get_topic_color(t) for t in bt_counts.index
            ]

            fig_bt = go.Figure(
                data=[
                    go.Bar(
                        x=bt_counts.values,
                        y=[
                            f"{BankingTopicCategorizer.get_topic_icon(t)} {t}"
                            for t in bt_counts.index
                        ],
                        orientation="h",
                        marker=dict(
                            color=bt_colors, line=dict(color="black", width=1.5)
                        ),
                        text=[f"{v:,}" for v in bt_counts.values],
                        textposition="outside",
                        textfont=dict(size=11),
                    )
                ]
            )
            fig_bt.update_layout(
                title=f"<b>Phân bố chủ đề - {selected_bank_topic}</b>",
                title_font=dict(size=14, color="#1e40af"),
                height=400,
                margin=dict(l=250),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        with col_bt2:
            # Most negative topic for this bank
            bt_sentiment = (
                bank_topic_df.groupby("topic_primary")
                .agg(
                    count=("sentiment", "size"),
                    neg_pct=("sentiment", lambda x: (x == "negative").mean() * 100),
                    avg_rating=("score", "mean"),
                )
                .round(1)
                .sort_values("neg_pct", ascending=False)
            )

            st.markdown(
                f"**📋 Chi tiết sentiment theo chủ đề ({selected_bank_topic})**"
            )

            # Display as styled table
            bt_display = bt_sentiment.copy()
            bt_display.columns = ["Số reviews", "% Tiêu cực", "Rating TB"]
            bt_display.index.name = "Chủ đề"
            st.dataframe(
                bt_display.style.format(
                    {
                        "Số reviews": "{:,.0f}",
                        "% Tiêu cực": "{:.1f}%",
                        "Rating TB": "{:.2f}",
                    }
                )
                .background_gradient(subset=["% Tiêu cực"], cmap="Reds")
                .background_gradient(subset=["Rating TB"], cmap="RdYlGn"),
                use_container_width=True,
                height=380,
            )

        st.markdown("---")

        # ============================================
        # 3.7: BẢNG THỐNG KÊ CHỦ ĐỀ TỔNG HỢP
        # ============================================
        st.markdown("### 📋 Bảng thống kê tổng hợp chủ đề")

        topic_stats_df = topic_categorizer.get_topic_stats(df_topics)

        st.dataframe(
            topic_stats_df.style.format(
                {
                    "Số reviews": "{:,.0f}",
                    "% Tổng": "{:.1f}%",
                    "% Tích cực": "{:.1f}%",
                    "% Tiêu cực": "{:.1f}%",
                    "Rating TB": "{:.2f}",
                }
            )
            .background_gradient(subset=["% Tích cực"], cmap="RdYlGn")
            .background_gradient(subset=["% Tiêu cực"], cmap="Reds")
            .background_gradient(subset=["Số reviews"], cmap="Blues"),
            use_container_width=True,
        )

        # ============================================
        # 3.8: SAMPLE REVIEWS PER TOPIC
        # ============================================
        st.markdown("---")
        st.markdown("### 📝 Reviews mẫu theo chủ đề")

        selected_topic_detail = st.selectbox(
            "Chọn chủ đề để xem reviews mẫu:",
            topic_categorizer.TOPIC_ORDER + ["Khác"],
            key="topic_detail_select",
        )

        topic_detail_df = df_topics[df_topics["topic_primary"] == selected_topic_detail]

        if len(topic_detail_df) > 0:
            col_td1, col_td2 = st.columns(2)

            with col_td1:
                st.markdown("**😊 Reviews tích cực:**")
                pos_samples = topic_detail_df[
                    topic_detail_df["sentiment"] == "positive"
                ].head(5)
                for _, row in pos_samples.iterrows():
                    with st.expander(
                        f"⭐ {row['score']} - {row.get('bank_name', 'N/A')}"
                    ):
                        st.write(str(row.get("content", "N/A"))[:300])

            with col_td2:
                st.markdown("**😞 Reviews tiêu cực:**")
                neg_samples = topic_detail_df[
                    topic_detail_df["sentiment"] == "negative"
                ].head(5)
                for _, row in neg_samples.iterrows():
                    with st.expander(
                        f"⭐ {row['score']} - {row.get('bank_name', 'N/A')}"
                    ):
                        st.write(str(row.get("content", "N/A"))[:300])
        else:
            st.info(f"Không có reviews nào thuộc chủ đề '{selected_topic_detail}'")

    # ========================================
    # TAB 4: SENTIMENT × CỔ PHIẾU (FINANCIAL)
    # ========================================
    with tab4:
        st.header("💹 Phân tích tương quan Sentiment × Giá cổ phiếu")
        st.markdown(
            """
        > **Câu hỏi nghiên cứu**: *Sentiment trên app store có tương quan/ảnh hưởng đến giá cổ phiếu ngân hàng không?*
        >
        > Dữ liệu cổ phiếu 11 ngân hàng niêm yết (HOSE), aggregate theo **tuần** để giảm noise.
        > *(Agribank: chưa niêm yết → loại khỏi phân tích)*
        """
        )

        # --- Load financial data ---
        @st.cache_data
        def load_financial_data():
            try:
                merged = pd.read_csv("./output/sentiment_stock_weekly.csv")
                merged["week"] = pd.to_datetime(merged["week"])
                return merged
            except FileNotFoundError:
                return None

        @st.cache_data
        def load_stock_data():
            try:
                stock = pd.read_csv("./output/stock_prices.csv")
                stock["date"] = pd.to_datetime(stock["date"])
                return stock
            except FileNotFoundError:
                return None

        @st.cache_data
        def load_correlation_data():
            try:
                return pd.read_csv("./output/correlation_results.csv")
            except FileNotFoundError:
                return None

        @st.cache_data
        def load_granger_data():
            try:
                return pd.read_csv("./output/granger_causality.csv")
            except FileNotFoundError:
                return None

        fin_merged = load_financial_data()
        fin_stock = load_stock_data()
        fin_corr = load_correlation_data()
        fin_granger = load_granger_data()

        if fin_merged is not None and len(fin_merged) > 0:
            # --- METRICS ---
            st.markdown("### 📊 Tổng quan dữ liệu tài chính")
            fm1, fm2, fm3, fm4 = st.columns(4)
            with fm1:
                st.metric("🏦 Ngân hàng", f"{fin_merged['bank_name'].nunique()}")
            with fm2:
                st.metric("📅 Số tuần", f"{fin_merged['week'].nunique()}")
            with fm3:
                st.metric("🔗 Data points", f"{len(fin_merged):,}")
            with fm4:
                if fin_corr is not None:
                    sig = fin_corr["significant_pearson"].sum()
                    st.metric("✅ Significant (p<0.05)", f"{sig}/{len(fin_corr)}")
                else:
                    st.metric("📐 Correlation", "N/A")

            st.markdown("---")

            # ============================================
            # 4.1: DUAL-AXIS CHART - Sentiment vs Stock Price
            # ============================================
            st.markdown("### 📈 Biến động Sentiment & Giá cổ phiếu theo tuần")

            fin_bank_select = st.selectbox(
                "Chọn ngân hàng:",
                sorted(fin_merged["bank_name"].unique()),
                key="fin_bank_select",
            )

            bank_fin = fin_merged[
                fin_merged["bank_name"] == fin_bank_select
            ].sort_values("week")

            if len(bank_fin) > 0:
                fig_dual = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=(
                        f"Giá cổ phiếu {fin_bank_select} (đóng cửa TB tuần)",
                        f"Sentiment {fin_bank_select} (Rating TB & % Tiêu cực)",
                    ),
                    row_heights=[0.5, 0.5],
                )

                # Row 1: Stock price
                fig_dual.add_trace(
                    go.Scatter(
                        x=bank_fin["week"],
                        y=bank_fin["avg_close"],
                        name="Giá CP (VNĐ)",
                        line=dict(color="#2563eb", width=2.5),
                        mode="lines+markers",
                        marker=dict(size=4),
                    ),
                    row=1,
                    col=1,
                )

                # Row 2: Sentiment metrics
                fig_dual.add_trace(
                    go.Scatter(
                        x=bank_fin["week"],
                        y=bank_fin["avg_rating"],
                        name="Rating TB",
                        line=dict(color="#10b981", width=2),
                        mode="lines+markers",
                        marker=dict(size=4),
                    ),
                    row=2,
                    col=1,
                )

                fig_dual.add_trace(
                    go.Bar(
                        x=bank_fin["week"],
                        y=bank_fin["negative_ratio"],
                        name="% Tiêu cực",
                        marker=dict(color="rgba(239,68,68,0.4)"),
                    ),
                    row=2,
                    col=1,
                )

                fig_dual.update_layout(
                    height=600,
                    title_font=dict(size=16, color="#1e40af"),
                    hovermode="x unified",
                    showlegend=True,
                    legend=dict(x=0, y=1.15, orientation="h"),
                )
                fig_dual.update_yaxes(title_text="Giá (VNĐ)", row=1, col=1)
                fig_dual.update_yaxes(title_text="Rating / %", row=2, col=1)

                st.plotly_chart(fig_dual, use_container_width=True)

            st.markdown("---")

            # ============================================
            # 4.2: SCATTER PLOT - Rating vs Return
            # ============================================
            st.markdown("### 🔍 Scatter: Rating TB vs Return tuần (%)")

            fig_scatter = px.scatter(
                fin_merged,
                x="avg_rating",
                y="weekly_return",
                color="bank_name",
                size="review_count",
                hover_data=["week", "avg_close"],
                title="<b>Rating trung bình vs Return tuần (%) - Tất cả ngân hàng</b>",
                labels={
                    "avg_rating": "Rating trung bình",
                    "weekly_return": "Return tuần (%)",
                    "bank_name": "Ngân hàng",
                    "review_count": "Số reviews",
                },
                trendline="ols",
            )
            fig_scatter.update_layout(
                height=500,
                title_font=dict(size=16, color="#1e40af"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Second scatter: Negative ratio vs Return
            st.markdown("### 🔍 Scatter: % Tiêu cực vs Return tuần (%)")

            fig_scatter2 = px.scatter(
                fin_merged,
                x="negative_ratio",
                y="weekly_return",
                color="bank_name",
                size="review_count",
                hover_data=["week", "avg_close"],
                title="<b>% Reviews tiêu cực vs Return tuần (%) - Tất cả ngân hàng</b>",
                labels={
                    "negative_ratio": "% Tiêu cực",
                    "weekly_return": "Return tuần (%)",
                    "bank_name": "Ngân hàng",
                    "review_count": "Số reviews",
                },
                trendline="ols",
            )
            fig_scatter2.update_layout(
                height=500,
                title_font=dict(size=16, color="#1e40af"),
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)

            st.markdown("---")

            # ============================================
            # 4.3: CORRELATION HEATMAP
            # ============================================
            if fin_corr is not None and len(fin_corr) > 0:
                st.markdown(
                    "### 📐 Heatmap tương quan Pearson (Rating TB ↔ Return tuần)"
                )

                # Filter for avg_rating vs weekly_return
                corr_rating_return = fin_corr[
                    (fin_corr["variable_x"] == "avg_rating")
                    & (fin_corr["variable_y"] == "weekly_return")
                ]

                if len(corr_rating_return) > 0:
                    # Create summary table for all correlation pairs
                    corr_pivot = fin_corr.pivot_table(
                        index="bank_name",
                        columns="label",
                        values="pearson_r",
                        aggfunc="first",
                    )

                    fig_corr_heat = go.Figure(
                        data=go.Heatmap(
                            z=corr_pivot.values,
                            x=corr_pivot.columns.tolist(),
                            y=corr_pivot.index.tolist(),
                            colorscale=[
                                [0, "#ef4444"],
                                [0.5, "#fef3c7"],
                                [1, "#10b981"],
                            ],
                            zmid=0,
                            text=corr_pivot.values.round(3),
                            texttemplate="%{text}",
                            textfont=dict(size=11, color="black"),
                            colorbar=dict(title="Pearson r"),
                        )
                    )
                    fig_corr_heat.update_layout(
                        title="<b>Hệ số tương quan Pearson theo ngân hàng</b>",
                        title_font=dict(size=16, color="#1e40af"),
                        height=500,
                        margin=dict(l=120, b=150),
                        xaxis=dict(tickangle=-30),
                    )
                    st.plotly_chart(fig_corr_heat, use_container_width=True)

                st.markdown("---")

                # ============================================
                # 4.4: CORRELATION TABLE
                # ============================================
                st.markdown("### 📋 Bảng kết quả tương quan chi tiết")

                corr_display = fin_corr[
                    [
                        "bank_name",
                        "label",
                        "n_weeks",
                        "pearson_r",
                        "pearson_p",
                        "spearman_r",
                        "spearman_p",
                        "significant_pearson",
                    ]
                ].copy()
                corr_display.columns = [
                    "Ngân hàng",
                    "Cặp biến",
                    "Số tuần",
                    "Pearson r",
                    "Pearson p",
                    "Spearman r",
                    "Spearman p",
                    "Có ý nghĩa?",
                ]

                st.dataframe(
                    corr_display.style.format(
                        {
                            "Pearson r": "{:.4f}",
                            "Pearson p": "{:.4f}",
                            "Spearman r": "{:.4f}",
                            "Spearman p": "{:.4f}",
                        }
                    ).apply(
                        lambda x: ["background-color: #d1fae5" if v else "" for v in x],
                        subset=["Có ý nghĩa?"],
                    ),
                    use_container_width=True,
                    height=400,
                )

                # Summary insight
                sig_pearson = fin_corr["significant_pearson"].sum()
                total_pairs = len(fin_corr)
                st.info(
                    f"""
                **📊 Tổng kết tương quan:**
                - Tổng cặp phân tích: **{total_pairs}**
                - Có ý nghĩa thống kê (p < 0.05): **{sig_pearson}** ({sig_pearson/total_pairs*100:.1f}%)
                - Không có ý nghĩa: **{total_pairs - sig_pearson}** ({(total_pairs-sig_pearson)/total_pairs*100:.1f}%)
                """
                )

            st.markdown("---")

            # ============================================
            # 4.5: GRANGER CAUSALITY RESULTS
            # ============================================
            if fin_granger is not None and len(fin_granger) > 0:
                st.markdown("### 🔬 Granger Causality Test")
                st.markdown(
                    """
                > **Granger Causality**: Kiểm tra xem sentiment **tuần trước** có giúp dự đoán return **tuần sau** không?
                > - ✅ **p < 0.05**: Sentiment CÓ khả năng dự báo (Granger-cause) stock return
                > - ❌ **p ≥ 0.05**: Không có bằng chứng về quan hệ nhân quả
                """
                )

                granger_display = fin_granger[
                    [
                        "bank_name",
                        "label",
                        "best_lag",
                        "p_value",
                        "significant",
                        "interpretation",
                    ]
                ].copy()
                granger_display.columns = [
                    "Ngân hàng",
                    "Kiểm định",
                    "Lag tốt nhất",
                    "p-value",
                    "Có ý nghĩa?",
                    "Diễn giải",
                ]

                st.dataframe(
                    granger_display.style.format(
                        {
                            "p-value": "{:.4f}",
                        }
                    ).apply(
                        lambda x: ["background-color: #d1fae5" if v else "" for v in x],
                        subset=["Có ý nghĩa?"],
                    ),
                    use_container_width=True,
                )

                sig_granger = fin_granger["significant"].sum()
                st.info(
                    f"""
                **🔬 Tổng kết Granger Causality:**
                - Có nhân quả Granger: **{sig_granger}/{len(fin_granger)}** cặp
                - Diễn giải: {"Một số ngân hàng cho thấy sentiment tuần trước có khả năng dự báo return tuần sau." if sig_granger > 0 else "Sentiment không Granger-cause stock return cho hầu hết ngân hàng — phù hợp với lý thuyết thị trường hiệu quả (EMH)."}
                """
                )

            st.markdown("---")

            # ============================================
            # 4.6: KEY FINDINGS
            # ============================================
            st.markdown("### 💡 Kết luận & Insights")

            avg_corr = (
                fin_corr["pearson_r"].mean()
                if fin_corr is not None and len(fin_corr) > 0
                else 0
            )

            if abs(avg_corr) < 0.1:
                finding = "**Không có tương quan có ý nghĩa** giữa sentiment trên app store và giá cổ phiếu ngân hàng."
                explain = """Điều này phù hợp với thực tế vì:
                1. Giá cổ phiếu bị ảnh hưởng bởi nhiều yếu tố vĩ mô (lãi suất, GDP, VN-Index...)
                2. Reviews app chỉ phản ánh trải nghiệm người dùng, không phải hiệu quả kinh doanh
                3. Số lượng reviews/tuần/bank còn hạn chế (~7-15 reviews)
                4. Phù hợp với **Giả thuyết Thị trường Hiệu quả (EMH)** — thông tin từ reviews đã được phản ánh trong giá
                """
            elif avg_corr > 0.1:
                finding = "**Có tương quan dương yếu** giữa sentiment và giá cổ phiếu."
                explain = "Ngân hàng có sentiment tốt hơn có xu hướng có hiệu suất cổ phiếu tốt hơn, nhưng tương quan yếu."
            else:
                finding = "**Có tương quan âm yếu** — cần phân tích sâu hơn."
                explain = "Kết quả ngược trực giác, có thể do các yếu tố nhiễu."

            st.success(
                f"""
            📊 {finding}

            {explain}

            > *Lưu ý: Correlation ≠ Causation. Kết quả này là phân tích khám phá (exploratory), không phải kết luận nhân quả.*
            """
            )

        else:
            st.warning(
                """
            ⚠️ **Chưa có dữ liệu tài chính.**

            Chạy script sau để tạo dữ liệu:
            ```bash
            python financial_analysis.py
            ```

            Script sẽ tự động:
            1. Crawl giá cổ phiếu 11 ngân hàng từ Yahoo Finance
            2. Merge với sentiment theo tuần
            3. Tính tương quan Pearson/Spearman
            4. Chạy Granger Causality Test
            """
            )

    # ========================================
    # TAB 5: SO SÁNH MÔ HÌNH ML
    # ========================================
    with tab5:
        st.header("🤖 So sánh các mô hình Machine Learning")

        if model_comparison is not None:
            model_comparison = model_comparison.set_index("Unnamed: 0")

            # Metrics của các mô hình
            st.markdown("### 📊 Hiệu suất các mô hình")

            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

            fig = go.Figure()

            # Extended color palette for 6+ models
            colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

            for idx, model_name in enumerate(model_comparison.index):
                values = [model_comparison.loc[model_name, m] for m in metrics]
                fig.add_trace(
                    go.Bar(
                        name=model_name,
                        x=metrics,
                        y=values,
                        marker=dict(
                            color=colors[idx % len(colors)],
                            line=dict(color="black", width=2),
                        ),
                        text=[f"{v:.4f}" for v in values],
                        textposition="outside",
                        textfont=dict(size=11),
                    )
                )

            fig.update_layout(
                title="<b>So sánh hiệu suất các mô hình</b>",
                title_font=dict(size=18, color="#1e40af"),
                xaxis_title="Metrics",
                yaxis_title="Score",
                height=500,
                barmode="group",
                yaxis=dict(range=[0.5, 1.05]),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Thời gian training/prediction
            st.markdown("### ⏱️ Thời gian xử lý")

            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()

                train_times = model_comparison["Training Time"]

                fig.add_trace(
                    go.Bar(
                        x=model_comparison.index,
                        y=train_times,
                        marker=dict(color="#fb923c", line=dict(color="black", width=2)),
                        text=[f"{t:.3f}s" for t in train_times],
                        textposition="outside",
                        textfont=dict(size=12, family="Arial Black"),
                    )
                )

                fig.update_layout(
                    title="<b>Training Time</b>",
                    title_font=dict(size=16, color="#1e40af"),
                    xaxis_title="Model",
                    yaxis_title="Time (seconds)",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()

                pred_times = model_comparison["Prediction Time"]

                fig.add_trace(
                    go.Bar(
                        x=model_comparison.index,
                        y=pred_times,
                        marker=dict(color="#a78bfa", line=dict(color="black", width=2)),
                        text=[f"{t:.4f}s" for t in pred_times],
                        textposition="outside",
                        textfont=dict(size=12, family="Arial Black"),
                    )
                )

                fig.update_layout(
                    title="<b>Prediction Time</b>",
                    title_font=dict(size=16, color="#1e40af"),
                    xaxis_title="Model",
                    yaxis_title="Time (seconds)",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

            # Bảng so sánh chi tiết
            st.markdown("### 📋 Bảng so sánh chi tiết")
            st.dataframe(
                model_comparison.style.format(
                    {
                        "Accuracy": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "Training Time": "{:.4f}s",
                        "Prediction Time": "{:.4f}s",
                    }
                )
                .highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                    color="lightgreen",
                )
                .highlight_min(
                    subset=["Training Time", "Prediction Time"], color="lightblue"
                ),
                use_container_width=True,
            )

            # Mô hình tốt nhất
            if metadata:
                st.markdown("### 🏆 Mô hình tốt nhất")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Model", metadata["model_name"])
                with col2:
                    st.metric("F1-Score", f"{metadata['f1_score']:.4f}")
                with col3:
                    st.metric("Accuracy", f"{metadata['accuracy']:.4f}")
                with col4:
                    st.metric("Features", f"{metadata['num_features']:,}")
        else:
            st.warning("⚠️ Chưa có dữ liệu so sánh mô hình")

    # ========================================
    # TAB 6: DỰ ĐOÁN SENTIMENT
    # ========================================
    with tab6:
        st.header("🔮 Dự đoán Sentiment cho Review mới")

        if model and tfidf_word:
            st.markdown(
                """
            <div class="prediction-box">
                <h3 style="color: white; text-align: center;">
                    ✨ Nhập review để phân tích cảm xúc ✨
                </h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Input review
            user_input = st.text_area(
                "📝 Nhập nội dung review:",
                height=150,
                placeholder="Ví dụ: App rất tốt, giao dịch nhanh chóng và an toàn...",
            )

            col1, col2, col3 = st.columns([1, 1, 1])

            with col2:
                predict_button = st.button(
                    "🚀 Dự đoán", use_container_width=True, type="primary"
                )

            if predict_button and user_input:
                with st.spinner("Đang phân tích..."):
                    (
                        sentiment,
                        confidence,
                        text_cleaned,
                        text_segmented,
                        has_real_confidence,
                    ) = predict_sentiment(user_input, model, tfidf_word, tfidf_char)

                    if sentiment:
                        st.markdown("---")
                        st.markdown("### 📊 Kết quả phân tích")

                        col1, col2 = st.columns(2)

                        with col1:
                            if "Tích cực" in sentiment:
                                st.success(f"### {sentiment}")
                                st.markdown(
                                    """
                                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                            padding: 20px; border-radius: 10px; text-align: center;">
                                    <h2 style="color: white; margin: 0;">😊 TÍCH CỰC</h2>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.error(f"### {sentiment}")
                                st.markdown(
                                    """
                                <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                                            padding: 20px; border-radius: 10px; text-align: center;">
                                    <h2 style="color: white; margin: 0;">😞 TIÊU CỰC</h2>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                        with col2:
                            # Always show confidence (use default if None)
                            display_confidence = (
                                confidence if confidence is not None else 75.0
                            )
                            st.metric("🎯 Độ tin cậy", f"{display_confidence:.2f}%")

                            # Gauge chart
                            fig = go.Figure(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=display_confidence,
                                    domain={"x": [0, 1], "y": [0, 1]},
                                    title={"text": "Confidence Score"},
                                    gauge={
                                        "axis": {"range": [None, 100]},
                                        "bar": {
                                            "color": (
                                                "#10b981"
                                                if "Tích cực" in sentiment
                                                else "#ef4444"
                                            )
                                        },
                                        "steps": [
                                            {"range": [0, 50], "color": "#fee2e2"},
                                            {"range": [50, 75], "color": "#fef3c7"},
                                            {"range": [75, 100], "color": "#d1fae5"},
                                        ],
                                        "threshold": {
                                            "line": {"color": "red", "width": 4},
                                            "thickness": 0.75,
                                            "value": 90,
                                        },
                                    },
                                )
                            )

                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                            # Show confidence type
                            if not has_real_confidence:
                                st.caption(
                                    "⚠️ Model không hỗ trợ predict_proba. Hiển thị độ tin cậy mặc định (75%)."
                                )
                            else:
                                st.caption(
                                    "✅ Độ tin cậy được tính từ model predict_proba."
                                )

                        # Hiển thị text đã xử lý (DEBUG)
                        st.markdown("### 🔍 Chi tiết xử lý text")

                        col_debug1, col_debug2 = st.columns(2)

                        with col_debug1:
                            st.info(f"❓ **Text gốc:**\n```\n{user_input}\n```")
                            st.info(
                                f"🧹 **Sau clean_text():**\n```\n{text_cleaned}\n```"
                            )

                        with col_debug2:
                            st.info(
                                f"🔤 **Sau word_tokenize():**\n```\n{text_segmented}\n```"
                            )

                            # Check for negation tokens
                            if (
                                "POSNEG_" in text_segmented
                                or "NEGNEG_" in text_segmented
                            ):
                                st.success("✅ **Đã xử lý phủ định!**")
                                if "POSNEG_" in text_segmented:
                                    st.markdown(
                                        "🟢 **POSNEG** token = Phủ định từ tiêu cực → Tích cực"
                                    )
                                if "NEGNEG_" in text_segmented:
                                    st.markdown(
                                        "🔴 **NEGNEG** token = Phủ định từ tích cực → Tiêu cực"
                                    )
                            else:
                                st.warning("⚠️ Không có token phủ định nào")

                        # Check if model was trained with negation
                        if metadata:
                            if any(
                                "POSNEG" in opt or "NEGNEG" in opt or "Negation" in opt
                                for opt in metadata.get("optimizations", [])
                            ):
                                st.info("✅ Model đã được train với negation handling")
                            else:
                                st.error(
                                    "⚠️ Model chưa được train với negation handling! Vui lòng retrain model trên Kaggle."
                                )

                        # 🔥 DEBUG: Check TF-IDF vocabulary
                        st.markdown("### 🔬 Debug TF-IDF Vocabulary")
                        tokens_in_text = text_segmented.split()
                        vocab_word = (
                            tfidf_word.vocabulary_
                            if hasattr(tfidf_word, "vocabulary_")
                            else {}
                        )
                        vocab_char = (
                            tfidf_char.vocabulary_
                            if hasattr(tfidf_char, "vocabulary_")
                            else {}
                        )

                        col_vocab1, col_vocab2 = st.columns(2)
                        with col_vocab1:
                            st.markdown("**🔤 Tokens trong text:**")
                            for token in tokens_in_text[:10]:
                                in_word_vocab = token in vocab_word
                                in_char_vocab = any(
                                    token[i : i + n] in vocab_char
                                    for n in range(2, 5)
                                    for i in range(len(token) - n + 1)
                                )

                                if in_word_vocab:
                                    st.success(f"✅ `{token}` → Word vocab")
                                elif in_char_vocab:
                                    st.warning(f"⚠️ `{token}` → Chỉ char vocab")
                                else:
                                    st.error(f"❌ `{token}` → KHÔNG có!")

                        with col_vocab2:
                            st.markdown("**📊 Feature Vector:**")
                            from scipy.sparse import hstack

                            text_word_vec = tfidf_word.transform([text_segmented])
                            text_char_vec = tfidf_char.transform([text_segmented])
                            text_combined = hstack([text_word_vec, text_char_vec])

                            st.info(f"Word: {text_word_vec.nnz} non-zero")
                            st.info(f"Char: {text_char_vec.nnz} non-zero")
                            st.info(f"Total: {text_combined.nnz} features")

                            if text_combined.nnz == 0:
                                st.error("⚠️ Vector rỗng! Model không nhận được gì!")

            elif predict_button and not user_input:
                st.warning("⚠️ Vui lòng nhập nội dung review!")

            # Ví dụ mẫu
            st.markdown("---")
            st.markdown("### 💡 Ví dụ mẫu")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**😊 Review tích cực:**")
                st.success(
                    "App rất tốt, giao diện đẹp, sử dụng dễ dàng. Giao dịch nhanh chóng và an toàn."
                )

            with col2:
                st.markdown("**😞 Review tiêu cực:**")
                st.error(
                    "App hay bị lỗi, đăng nhập không được. Giao dịch chậm và không ổn định."
                )
        else:
            st.warning(
                "⚠️ Chưa load được mô hình ML. Vui lòng kiểm tra file best_model_advanced.pkl và tfidf_word_vectorizer.pkl, tfidf_char_vectorizer.pkl"
            )

    # ========================================
    # TAB 7: DỮ LIỆU CHI TIẾT
    # ========================================
    with tab7:
        st.header("📝 Dữ liệu Reviews chi tiết")

        # Thống kê
        st.markdown(f"### Hiển thị {len(df_filtered):,} reviews")

        # Chọn cột hiển thị
        available_columns = df_filtered.columns.tolist()
        default_columns = ["bank_name", "content", "score", "sentiment_vi", "at"]
        default_columns = [col for col in default_columns if col in available_columns]

        selected_columns = st.multiselect(
            "Chọn cột hiển thị:", available_columns, default=default_columns
        )

        if selected_columns:
            # Hiển thị dataframe
            display_df = df_filtered[selected_columns].copy()

            # Format datetime
            if "at" in display_df.columns:
                display_df["at"] = pd.to_datetime(display_df["at"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

            st.dataframe(display_df, use_container_width=True, height=600)

            # Download button
            csv = df_filtered[selected_columns].to_csv(
                index=False, encoding="utf-8-sig"
            )
            st.download_button(
                label="📥 Tải xuống dữ liệu (CSV)",
                data=csv,
                file_name=f"reviews_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("📌 Vui lòng chọn ít nhất một cột để hiển thị")

        # Top reviews
        st.markdown("---")
        st.markdown("### 🔝 Top Reviews")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**😊 Top 5 Reviews Tích Cực (Rating cao nhất)**")
            top_positive = df_filtered[df_filtered["sentiment"] == "positive"].nlargest(
                5, "score"
            )
            for idx, row in top_positive.iterrows():
                with st.expander(f"⭐ {row['score']} - {row.get('bank_name', 'N/A')}"):
                    st.write(row.get("content", "N/A")[:300] + "...")

        with col2:
            st.markdown("**😞 Top 5 Reviews Tiêu Cực (Rating thấp nhất)**")
            top_negative = df_filtered[
                df_filtered["sentiment"] == "negative"
            ].nsmallest(5, "score")
            for idx, row in top_negative.iterrows():
                with st.expander(f"⭐ {row['score']} - {row.get('bank_name', 'N/A')}"):
                    st.write(row.get("content", "N/A")[:300] + "...")

else:
    st.error(
        "❌ Không thể load dữ liệu. Vui lòng kiểm tra file reviews_with_sentiment.csv"
    )
    st.info("💡 Hướng dẫn: Chạy file kltn.py trên Google Colab để tạo dữ liệu trước.")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p><strong>🎓 Đồ án tốt nghiệp: Phân tích cảm xúc người dùng ứng dụng ngân hàng di động</strong></p>
    <p>🔧 Công nghệ: Python, Machine Learning, Streamlit, Plotly</p>
</div>
""",
    unsafe_allow_html=True,
)
