# -*- coding: utf-8 -*-
"""
SENTIMENT ANALYSIS - MOBILE BANKING APPLICATIONS (KAGGLE VERSION)
==================================================================
Phân tích cảm xúc người dùng ứng dụng ngân hàng di động tại Việt Nam
"""

# ========================================
# 0. CÀI ĐẶT THƯ VIỆN (CHỈ CÀI THIẾU)
# ========================================
import sys
import subprocess

def install_if_missing(package):
    """Cài đặt package nếu chưa có"""
    try:
        __import__(package.split('[')[0])
    except ImportError:
        print(f"Đang cài đặt {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Cài đặt các thư viện cần thiết (không có sẵn trên Kaggle)
packages = [
    'google-play-scraper',
    'underthesea',
    'wordcloud',
    'imbalanced-learn'
]

for pkg in packages:
    install_if_missing(pkg)

print("✅ Đã cài đặt tất cả thư viện cần thiết!\n")

# ========================================
# 1. IMPORT THƯ VIỆN
# ========================================
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import os
import pickle
import time
from typing import Dict, List

# Scraping
from google_play_scraper import app
from google_play_scraper.features.reviews import reviews, Sort

# NLP
from underthesea import word_tokenize

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
from wordcloud import WordCloud

# Kaggle paths
KAGGLE_WORKING_DIR = '/kaggle/working'
OUTPUT_DIR = '/kaggle/working/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Import thành công tất cả thư viện!")
print(f"📁 Output directory: {OUTPUT_DIR}\n")

# ========================================
# 2. DANH SÁCH APP NGÂN HÀNG
# ========================================
APP_PACKAGES = [
    'com.vnpay.bidv',
    'com.vnpay.Agribank3g',
    'com.VCB',
    'com.sacombank.ewallet',
    'com.mbmobile',
    'com.vietinbank.ipay',
    'com.tpb.mb.gprsandroid',
    'vn.com.techcombank.bb.app',
    'com.vnpay.hdbank',
    'mobile.acb.com.vn',
    'vn.com.ocb.awe',
    'vn.com.msb.smartBanking',
]

APP_NAMES = {
    'com.vnpay.bidv': 'BIDV',
    'com.vnpay.Agribank3g': 'Agribank',
    'com.VCB': 'Vietcombank',
    'com.sacombank.ewallet': 'Sacombank',
    'com.mbmobile': 'MBBank',
    'com.vietinbank.ipay': 'VietinBank',
    'com.tpb.mb.gprsandroid': 'TPBank',
    'vn.com.techcombank.bb.app': 'Techcombank',
    'com.vnpay.hdbank': 'HDBank',
    'mobile.acb.com.vn': 'ACB',
    'vn.com.ocb.awe': 'OCB',
    'vn.com.msb.smartBanking': 'MSB'
}

print(f"✅ Danh sách {len(APP_PACKAGES)} ngân hàng:\n")
for pkg, name in APP_NAMES.items():
    print(f"   • {name}")
print()

# ========================================
# 3. CLASS VIETNAMESE REVIEW CLEANER
# ========================================
class VietnameseReviewCleaner:
    """
    Bộ xử lý text cho reviews tiếng Việt
    - Chuẩn hóa teencode
    - Sửa viết tắt
    - Loại bỏ icon/emoji
    - Sửa lỗi chính tả phổ biến
    """

    def __init__(self):
        # Từ điển teencode -> chuẩn
        self.teencode_dict = {
            'k': 'không', 'ko': 'không', 'hok': 'không', 'hong': 'không',
            'hem': 'không', 'kg': 'không', 'kh': 'không', 'khong': 'không',
            'dc': 'được', 'đc': 'được', 'dk': 'được', 'đk': 'được',
            'vs': 'với', 'vc': 'với', 'v': 'với',
            'ms': 'mới', 'mik': 'mình', 'mk': 'mình', 'mh': 'mình', 'mjh': 'mình',
            'tui': 'tôi', 'toy': 'tôi', 'toj': 'tôi',
            'bt': 'bình thường', 'bth': 'bình thường',
            'ntn': 'như thế nào', 'sao': 'như thế nào',
            'r': 'rồi', 'rùi': 'rồi', 'rui': 'rồi',
            'ak': 'à', 'ạk': 'ạ', 'nhaa': 'nhé', 'nha': 'nhé',
            'nek': 'nè', 'né': 'nè',
            'cx': 'cũng', 'cug': 'cũng',
            'bik': 'biết', 'bit': 'biết', 'bjt': 'biết',
            'z': 'vậy', 'zay': 'vậy',
            'wa': 'quá', 'qá': 'quá', 'wá': 'quá',
            'j': 'gì', 'zì': 'gì', 'jì': 'gì',
            'chs': 'chưa', 'chx': 'chưa',
            'tks': 'cảm ơn', 'tk': 'cảm ơn', 'thanks': 'cảm ơn',
            'tnks': 'cảm ơn', 'cam on': 'cảm ơn',
            'nc': 'nói chuyện', 'nch': 'nói chuyện',
            'ngta': 'người ta', 'nguoi ta': 'người ta', 'nta': 'người ta',
            'uk': 'ừ', 'uh': 'ừ',
            'oke': 'ok', 'okie': 'ok', 'okee': 'ok',
            'đag': 'đang', 'dg': 'đang', 'dang': 'đang',
            'lm': 'làm', 'lam': 'làm',
            'pk': 'phải', 'fai': 'phải', 'pải': 'phải',
            'xl': 'xin lỗi', 'sr': 'xin lỗi', 'sorry': 'xin lỗi',
            'tl': 'trả lời', 'rep': 'trả lời',
            'ib': 'inbox', 'mess': 'nhắn tin',
            'sd': 'sử dụng', 'xd': 'sử dụng',
            'ng': 'người', 'nguoi': 'người',
            'bn': 'bạn', 'b': 'bạn',
            'mn': 'mọi người', 'ad': 'admin',
            'app': 'ứng dụng', 'ứg dụng': 'ứng dụng', 'ung dung': 'ứng dụng',
        }

        # Từ điển lỗi chính tả phổ biến
        self.typo_dict = {
            'duoc': 'được', 'nhu': 'như', 'chi': 'chỉ',
            'nhung': 'nhưng', 'ma': 'mà', 'thi': 'thì',
            'tai': 'tại', 'lai': 'lại', 'nua': 'nữa',
            'dau': 'đâu', 'khi': 'khi', 'rat': 'rất',
            'tot': 'tốt', 'xau': 'xấu', 'te': 'tệ',
            'hay': 'hay', 'den': 'đến', 'cho': 'cho',
            'chua': 'chưa', 'roi': 'rồi', 'vao': 'vào',
            'xem': 'xem', 'nhan': 'nhận', 'gui': 'gửi',
        }

        # Icon/emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )

    def clean_text(self, text: str) -> str:
        """Làm sạch text đầy đủ"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = self.emoji_pattern.sub('', text)
        text = re.sub(r'[^\w\s\.,!?áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{2,}', r'\1', text)

        words = text.split()
        words = [self.teencode_dict.get(w, w) for w in words]
        text = ' '.join(words)

        words = text.split()
        words = [self.typo_dict.get(w, w) for w in words]
        text = ' '.join(words)

        return ' '.join(text.split())

    def clean_dataframe(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """Làm sạch toàn bộ dataframe"""
        df_cleaned = df.copy()
        print(f"Đang làm sạch {len(df_cleaned):,} reviews...")
        df_cleaned[f'{text_column}_cleaned'] = df_cleaned[text_column].apply(self.clean_text)

        empty_reviews = df_cleaned[f'{text_column}_cleaned'].str.strip().eq('').sum()
        print(f"✅ Hoàn thành!")
        print(f"   Reviews trống: {empty_reviews:,}")
        print(f"   Reviews hợp lệ: {len(df_cleaned) - empty_reviews:,}\n")
        return df_cleaned

print("✅ Class VietnameseReviewCleaner đã được định nghĩa!\n")

# ========================================
# 4. CRAWL THÔNG TIN APP
# ========================================
print("="*80)
print("BƯỚC 1: THU THẬP THÔNG TIN ỨNG DỤNG")
print("="*80 + "\n")

app_infos = []

for ap in tqdm(APP_PACKAGES, desc="Crawling app info"):
    try:
        info = app(ap, lang="vi", country="vn")
        info.pop("comments", None)
        app_infos.append(info)
        time.sleep(0.5)  # Tránh rate limit
    except Exception as e:
        print(f"\n⚠️  Lỗi {ap}: {e}")

app_infos_df = pd.DataFrame(app_infos)
app_infos_df.to_csv(f"{OUTPUT_DIR}/apps_info.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ Đã thu thập thông tin {len(app_infos)} ứng dụng")
print(f"📁 Đã lưu: apps_info.csv\n")

# ========================================
# 5. CRAWL REVIEWS
# ========================================
print("="*80)
print("BƯỚC 2: THU THẬP REVIEWS")
print("="*80 + "\n")

app_reviews = []
total_crawled = 0

for ap in tqdm(APP_PACKAGES, desc="Crawling reviews"):
    try:
        for score in range(1, 6):
            for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
                try:
                    count = 200 if score == 3 else 100
                    data, _ = reviews(
                        ap,
                        lang="vi",
                        country="vn",
                        sort=sort_order,
                        count=count,
                        filter_score_with=score
                    )

                    for r in data:
                        r["appId"] = ap
                        r["sortOrder"] = (
                            "most_relevant" if sort_order == Sort.MOST_RELEVANT else "newest"
                        )

                    app_reviews.extend(data)
                    total_crawled += len(data)
                    time.sleep(1)  # Tránh rate limit

                except Exception as e:
                    print(f"\n⚠️  Lỗi crawl {ap} (score={score}): {e}")
                    continue

    except Exception as e:
        print(f"\n⚠️  Lỗi crawl {ap}: {e}")

app_reviews_df = pd.DataFrame(app_reviews)

# Loại bỏ duplicates
app_reviews_df = app_reviews_df.drop_duplicates(subset=['reviewId'], keep='first')

app_reviews_df.to_csv(f"{OUTPUT_DIR}/apps_reviews.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ Đã thu thập {len(app_reviews_df):,} reviews (sau khi loại bỏ trùng lặp)")
print(f"📁 Đã lưu: apps_reviews.csv\n")

# ========================================
# 6. CLEAN DATA
# ========================================
print("="*80)
print("BƯỚC 3: LÀM SẠCH DỮ LIỆU")
print("="*80 + "\n")

cleaner = VietnameseReviewCleaner()
df_cleaned = cleaner.clean_dataframe(app_reviews_df, text_column='content')

# Lưu file cleaned
df_cleaned.to_csv(f"{OUTPUT_DIR}/reviews_cleaned.csv", index=False, encoding='utf-8-sig')
print(f"📁 Đã lưu: reviews_cleaned.csv\n")

# ========================================
# 7. GẮN NHÃN SENTIMENT
# ========================================
print("="*80)
print("BƯỚC 4: GẮN NHÃN SENTIMENT")
print("="*80 + "\n")

print("📊 Phân loại: 1-3⭐ = TIÊU CỰC | 4-5⭐ = TÍCH CỰC\n")

df_cleaned['sentiment'] = df_cleaned['score'].apply(
    lambda x: 'negative' if x <= 3 else 'positive'
)
df_cleaned['sentiment_vi'] = df_cleaned['score'].apply(
    lambda x: 'tiêu cực' if x <= 3 else 'tích cực'
)
df_cleaned['sentiment_label'] = df_cleaned['score'].apply(
    lambda x: 0 if x <= 3 else 1
)

sentiment_counts = df_cleaned['sentiment_vi'].value_counts()
total = len(df_cleaned)

print("✅ Đã gắn nhãn!\n")
print("📊 THỐNG KÊ SENTIMENT:")
print("="*80)

for sentiment, count in sentiment_counts.items():
    percentage = (count / total) * 100
    emoji = "😊" if sentiment == "tích cực" else "😞"
    print(f"   {emoji} {sentiment.upper()}: {count:,} ({percentage:.1f}%)")

print("="*80 + "\n")

# Thêm tên ngân hàng
df_cleaned['bank_name'] = df_cleaned['appId'].map(APP_NAMES)

# ========================================
# 8. VẼ BIỂU ĐỒ PHÂN BỐ
# ========================================
print("="*80)
print("BƯỚC 5: TẠO VISUALIZATION")
print("="*80 + "\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ 1: Phân bố Sentiment
sentiment_counts = df_cleaned['sentiment_vi'].value_counts()
colors = ['#ff6b6b', '#51cf66']

bars1 = axes[0].bar(sentiment_counts.index, sentiment_counts.values, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_title('PHÂN BỐ SENTIMENT', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Sentiment', fontsize=12)
axes[0].set_ylabel('Số lượng reviews', fontsize=12)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars1:
    height = bar.get_height()
    percentage = (height / total) * 100
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Biểu đồ 2: Phân bố Rating
rating_counts = df_cleaned['score'].value_counts().sort_index()
colors_rating = ['#e03131', '#f76707', '#fab005', '#74b816', '#20c997']

bars2 = axes[1].bar(rating_counts.index, rating_counts.values,
                   color=colors_rating, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_title('PHÂN BỐ THEO RATING', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Số sao', fontsize=12)
axes[1].set_ylabel('Số lượng reviews', fontsize=12)
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars2:
    height = bar.get_height()
    percentage = (height / total) * 100
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Đã tạo biểu đồ phân bố sentiment\n")

# Lưu file với sentiment
df_cleaned.to_csv(f"{OUTPUT_DIR}/reviews_with_sentiment.csv", index=False, encoding='utf-8-sig')
print(f"📁 Đã lưu: reviews_with_sentiment.csv\n")

# ========================================
# 9. TÁCH TỪ TIẾNG VIỆT
# ========================================
print("="*80)
print("BƯỚC 6: TÁCH TỪ TIẾNG VIỆT")
print("="*80 + "\n")

print("Đang tách từ cho reviews...")
df_cleaned['content_segmented'] = df_cleaned['content_cleaned'].apply(
    lambda x: word_tokenize(x, format="text") if isinstance(x, str) and x.strip() else ""
)

print("✅ Hoàn thành tách từ!\n")

# Hiển thị ví dụ
samples = df_cleaned[df_cleaned['content_cleaned'].str.len() > 20].sample(n=min(3, len(df_cleaned)))
print("📝 Ví dụ tách từ:")
print("="*80)
for idx, row in enumerate(samples.itertuples(), 1):
    print(f"\n[{idx}] TRƯỚC: {row.content_cleaned[:80]}...")
    print(f"    SAU:   {row.content_segmented[:80]}...")
print("\n" + "="*80 + "\n")

# ========================================
# 10. CHUẨN BỊ DỮ LIỆU CHO ML
# ========================================
print("="*80)
print("BƯỚC 7: CHUẨN BỊ DỮ LIỆU CHO ML")
print("="*80 + "\n")

# Lọc reviews hợp lệ
df_ml = df_cleaned[
    (df_cleaned['content_segmented'].notna()) & 
    (df_cleaned['content_segmented'].str.strip() != '') &
    (df_cleaned['sentiment_label'].notna())
].copy()

print(f"✅ Số lượng reviews hợp lệ: {len(df_ml):,}\n")

# Tách train/test
X = df_ml['content_segmented']
y = df_ml['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train set: {len(X_train):,} reviews")
print(f"📊 Test set: {len(X_test):,} reviews\n")

# TF-IDF Vectorization
print("🔧 Đang vectorize text với TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"✅ TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"✅ Số lượng features: {len(tfidf.get_feature_names_out()):,}\n")

# ========================================
# 11. XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH ML
# ========================================
print("="*80)
print("BƯỚC 8: XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH ML")
print("="*80 + "\n")

results = {}
trained_models = {}

# Định nghĩa các mô hình
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'SVM (Linear)': LinearSVC(max_iter=2000, C=1.0, random_state=42, dual='auto')
}

# Train và đánh giá từng mô hình
for model_name, model in models.items():
    print(f"{'='*80}")
    print(f"🤖 TRAINING: {model_name}")
    print(f"{'='*80}")
    
    # Training
    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = model.predict(X_test_tfidf)
    pred_time = time.time() - start_time
    
    # Lưu model
    trained_models[model_name] = model
    
    # Tính toán metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training Time': train_time,
        'Prediction Time': pred_time
    }
    
    print(f"\n✅ KẾT QUẢ:")
    print(f"   • Accuracy:  {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall:    {recall:.4f}")
    print(f"   • F1-Score:  {f1:.4f}")
    print(f"   • Training time: {train_time:.2f}s")
    print(f"   • Prediction time: {pred_time:.4f}s\n")
    
    print(f"📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Tiêu cực', 'Tích cực'],
                               digits=4))
    
    print(f"📊 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
    print(f"    [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]\n")

print("="*80 + "\n")

# ========================================
# 12. SO SÁNH CÁC MÔ HÌNH
# ========================================
print("="*80)
print("BƯỚC 9: SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
print("="*80 + "\n")

# Tạo DataFrame so sánh
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)

print("📊 BẢNG SO SÁNH:")
print("="*80)
print(comparison_df.to_string())
print("="*80 + "\n")

# Tìm mô hình tốt nhất
best_model_name = comparison_df['F1-Score'].idxmax()
best_f1 = comparison_df.loc[best_model_name, 'F1-Score']

print(f"🏆 MÔ HÌNH TỐT NHẤT: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}\n")

# Vẽ biểu đồ so sánh
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: So sánh metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

colors = ['#3b82f6', '#ef4444', '#10b981']
for idx, (model_name, color) in enumerate(zip(results.keys(), colors)):
    values = [results[model_name][m] for m in metrics]
    axes[0].bar(x + idx*width, values, width, label=model_name, 
               color=color, alpha=0.8, edgecolor='black', linewidth=1.5)

axes[0].set_xlabel('Metrics', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics)
axes[0].legend(loc='lower right')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].set_ylim([0.5, 1.0])

# Biểu đồ 2: So sánh thời gian
model_names = list(results.keys())
train_times = [results[m]['Training Time'] for m in model_names]
pred_times = [results[m]['Prediction Time'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = axes[1].bar(x - width/2, train_times, width, label='Training Time',
                   color='#fb923c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = axes[1].bar(x + width/2, pred_times, width, label='Prediction Time',
                   color='#a78bfa', alpha=0.8, edgecolor='black', linewidth=1.5)

axes[1].set_xlabel('Models', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
axes[1].set_title('SO SÁNH THỜI GIAN', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=15, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s',
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Lưu kết quả
comparison_df.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', encoding='utf-8-sig')
print(f"\n📁 Đã lưu: model_comparison.csv\n")

# ========================================
# 13. PHÂN TÍCH THEO NGÂN HÀNG
# ========================================
print("="*80)
print("BƯỚC 10: PHÂN TÍCH THEO TỪNG NGÂN HÀNG")
print("="*80 + "\n")

# Thống kê theo ngân hàng
bank_stats = df_cleaned.groupby('bank_name').agg({
    'content': 'count',
    'score': 'mean',
    'sentiment_label': lambda x: (x == 1).sum() / len(x) * 100
}).round(2)

bank_stats.columns = ['Số reviews', 'Rating TB', '% Tích cực']
bank_stats = bank_stats.sort_values('Rating TB', ascending=False)

print("📊 THỐNG KÊ THEO NGÂN HÀNG:")
print("="*80)
print(bank_stats.to_string())
print("="*80 + "\n")

# Vẽ biểu đồ
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Biểu đồ 1: Rating trung bình
banks = bank_stats.index
ratings = bank_stats['Rating TB']
colors = ['#22c55e' if r >= 4 else '#eab308' if r >= 3 else '#ef4444' for r in ratings]

bars = axes[0].barh(banks, ratings, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Rating Trung Bình', fontsize=12, fontweight='bold')
axes[0].set_title('RATING TRUNG BÌNH THEO NGÂN HÀNG', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlim([0, 5])
axes[0].grid(axis='x', alpha=0.3, linestyle='--')

for bar, rating in zip(bars, ratings):
    axes[0].text(rating + 0.1, bar.get_y() + bar.get_height()/2,
                f'{rating:.2f}⭐',
                va='center', fontsize=10, fontweight='bold')

# Biểu đồ 2: % Tích cực
positive_pct = bank_stats['% Tích cực']
colors = ['#22c55e' if p >= 50 else '#ef4444' for p in positive_pct]

bars = axes[1].barh(banks, positive_pct, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('% Reviews Tích Cực', fontsize=12, fontweight='bold')
axes[1].set_title('TỶ LỆ REVIEWS TÍCH CỰC THEO NGÂN HÀNG', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlim([0, 100])
axes[1].grid(axis='x', alpha=0.3, linestyle='--')

for bar, pct in zip(bars, positive_pct):
    axes[1].text(pct + 2, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%',
                va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/bank_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Lưu thống kê
bank_stats.to_csv(f'{OUTPUT_DIR}/bank_statistics.csv', encoding='utf-8-sig')
print(f"📁 Đã lưu: bank_statistics.csv\n")

# ========================================
# 14. LƯU MÔ HÌNH TỐT NHẤT
# ========================================
print("="*80)
print("BƯỚC 11: LƯU MÔ HÌNH TỐT NHẤT")
print("="*80 + "\n")

best_model = trained_models[best_model_name]

# Lưu model
model_path = f'{OUTPUT_DIR}/best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"💾 Đã lưu model: {best_model_name}")

# Lưu vectorizer
vectorizer_path = f'{OUTPUT_DIR}/tfidf_vectorizer.pkl'
with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf, f)
print(f"💾 Đã lưu TF-IDF vectorizer")

# Lưu metadata
metadata = {
    'model_name': best_model_name,
    'f1_score': float(best_f1),
    'accuracy': float(results[best_model_name]['Accuracy']),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_features': len(tfidf.get_feature_names_out()),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}

metadata_path = f'{OUTPUT_DIR}/model_metadata.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"💾 Đã lưu metadata\n")

# ========================================
# 15. TẠO WORD CLOUD
# ========================================
print("="*80)
print("BƯỚC 12: TẠO WORD CLOUD")
print("="*80 + "\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Word Cloud cho reviews tiêu cực
negative_text = ' '.join(df_cleaned[df_cleaned['sentiment'] == 'negative']['content_segmented'].astype(str))

wordcloud_neg = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='Reds',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10
).generate(negative_text)

axes[0].imshow(wordcloud_neg, interpolation='bilinear')
axes[0].set_title('WORD CLOUD - REVIEWS TIÊU CỰC', fontsize=14, fontweight='bold', pad=15)
axes[0].axis('off')

# Word Cloud cho reviews tích cực
positive_text = ' '.join(df_cleaned[df_cleaned['sentiment'] == 'positive']['content_segmented'].astype(str))

wordcloud_pos = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='Greens',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10
).generate(positive_text)

axes[1].imshow(wordcloud_pos, interpolation='bilinear')
axes[1].set_title('WORD CLOUD - REVIEWS TÍCH CỰC', fontsize=14, fontweight='bold', pad=15)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Đã tạo word cloud\n")

# ========================================
# 16. LƯU FILE CHO ML
# ========================================
ml_cols = ['content_cleaned', 'content_segmented', 'sentiment', 'sentiment_label', 'score', 'bank_name']
df_ml_output = df_cleaned[[c for c in ml_cols if c in df_cleaned.columns]]
df_ml_output.to_csv(f'{OUTPUT_DIR}/reviews_for_ml.csv', index=False, encoding='utf-8-sig')
print(f"📁 Đã lưu: reviews_for_ml.csv\n")

# ========================================
# 17. TÓM TẮT HOÀN CHỈNH
# ========================================
print("="*80)
print("🎉 TÓM TẮT DỰ ÁN HOÀN CHỈNH")
print("="*80)

print(f"""
📊 DỮ LIỆU:
   • Tổng số reviews: {len(df_cleaned):,}
   • Số ngân hàng: {df_cleaned['appId'].nunique()}
   • Reviews tiêu cực: {len(df_cleaned[df_cleaned['sentiment']=='negative']):,}
   • Reviews tích cực: {len(df_cleaned[df_cleaned['sentiment']=='positive']):,}

🤖 MÔ HÌNH MACHINE LEARNING:
   • Mô hình đã training: {len(results)}
   • Mô hình tốt nhất: {best_model_name}
   • F1-Score: {best_f1:.4f}
   • Accuracy: {results[best_model_name]['Accuracy']:.4f}

📁 FILES ĐÃ TẠO TRONG {OUTPUT_DIR}:
   • apps_info.csv - Thông tin ứng dụng
   • apps_reviews.csv - Reviews gốc
   • reviews_cleaned.csv - Reviews đã làm sạch
   • reviews_with_sentiment.csv - Reviews có nhãn sentiment
   • reviews_for_ml.csv - Dữ liệu cho ML
   • model_comparison.csv - So sánh các mô hình
   • bank_statistics.csv - Thống kê theo ngân hàng
   • best_model.pkl - Mô hình ML tốt nhất
   • tfidf_vectorizer.pkl - TF-IDF vectorizer
   • model_metadata.json - Metadata của mô hình

📈 BIỂU ĐỒ:
   • sentiment_distribution.png
   • model_comparison.png
   • bank_comparison.png
   • wordcloud.png
""")

print("="*80)
print("✨ DỰ ÁN ĐÃ HOÀN THÀNH TRÊN KAGGLE!")
print("📥 Tất cả files đã được lưu trong /kaggle/working/output")
print("💡 Bạn có thể download tất cả files từ Output tab")
print("="*80)

# ========================================
# 18. KIỂM TRA CÁC FILE ĐÃ TẠO
# ========================================
print("\n📁 DANH SÁCH FILES ĐÃ TẠO:\n")
for file in sorted(os.listdir(OUTPUT_DIR)):
    file_path = os.path.join(OUTPUT_DIR, file)
    size = os.path.getsize(file_path)
    size_mb = size / (1024 * 1024)
    print(f"   ✅ {file} ({size_mb:.2f} MB)")

print("\n" + "="*80)
print("🎊 HOÀN TẤT! Dự án đã sẵn sàng để demo và nộp báo cáo!")
print("="*80)
