# 🚀 HƯỚNG DẪN CHẠY NHANH - QUICK START GUIDE

## 📋 Tóm tắt

Dự án gồm 2 phần chính:

1. **kltn.py** - Thu thập dữ liệu & Training models (chạy trên Google Colab)
2. **dashboard.py** - Web dashboard (chạy trên máy local)

---

## ⚡ CÁCH CHẠY NHANH NHẤT

### Bước 1: Chạy trên Google Colab (15-30 phút)

1. Truy cập: https://colab.research.google.com/
2. Upload file `kltn.py`
3. Chạy tất cả cells (Runtime → Run all)
4. Chờ hoàn thành
5. Download các file từ Google Drive:
   - `reviews_with_sentiment.csv`
   - `bank_statistics.csv`
   - `model_comparison.csv`
   - `best_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `model_metadata.json`

### Bước 2: Chạy Dashboard trên máy (2-3 phút)

**Windows:**

```cmd
# Đặt các file vào thư mục dự án
# Double click file: run_dashboard.bat
```

**macOS/Linux:**

```bash
# Đặt các file vào thư mục dự án
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Hoặc chạy thủ công:**

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy dashboard
streamlit run dashboard.py
```

3. Mở browser tại: `http://localhost:8501`

---

## 🎯 Chi tiết từng bước

### PHẦN 1: Google Colab (Thu thập & Xử lý dữ liệu)

#### Step 1: Setup môi trường

```python
# Cell 1: Cài đặt thư viện (chạy 1 lần duy nhất)
!pip install -qq google-play-scraper underthesea scikit-learn imbalanced-learn wordcloud
```

#### Step 2: Import và Mount Drive

```python
# Cell 2: Import thư viện
import pandas as pd
import numpy as np
# ... (auto có trong kltn.py)

# Cell 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

#### Step 3: Chạy pipeline (tự động)

- ✅ Crawl reviews từ 12 ngân hàng
- ✅ Làm sạch dữ liệu
- ✅ Gán nhãn sentiment
- ✅ Tách từ tiếng Việt
- ✅ Training 3 models (Naive Bayes, Logistic Regression, SVM)
- ✅ So sánh kết quả
- ✅ Tạo visualization

**Thời gian ước tính:**

- Crawl data: 15-20 phút
- Training models: 5-10 phút
- Visualization: 1-2 phút

#### Step 4: Kết quả

Files được tạo trong Google Drive:

```
/content/drive/MyDrive/google_play_scraper/
├── apps_info.csv                    # Thông tin 12 ngân hàng
├── apps_reviews.csv                 # ~10k-15k reviews gốc
├── reviews_cleaned.csv              # Reviews đã clean
├── reviews_with_sentiment.csv       # ⭐ FILE QUAN TRỌNG
├── reviews_for_ml.csv               # Data cho ML
├── bank_statistics.csv              # ⭐ FILE QUAN TRỌNG
├── model_comparison.csv             # ⭐ FILE QUAN TRỌNG
├── best_model.pkl                   # ⭐ MODEL
├── tfidf_vectorizer.pkl             # ⭐ VECTORIZER
├── model_metadata.json              # ⭐ METADATA
├── sentiment_distribution.png       # Biểu đồ
├── model_comparison.png             # Biểu đồ
├── bank_comparison.png              # Biểu đồ
└── wordcloud.png                    # Word cloud
```

**⭐ = Files bắt buộc cho dashboard**

---

### PHẦN 2: Local Dashboard (Web App)

#### Chuẩn bị

1. Tạo thư mục dự án:

```bash
mkdir sentiment_analysis_banking
cd sentiment_analysis_banking
```

2. Copy files:

```
sentiment_analysis_banking/
├── dashboard.py
├── requirements.txt
├── run_dashboard.sh (macOS/Linux)
├── run_dashboard.bat (Windows)
├── reviews_with_sentiment.csv      # Từ Google Drive
├── bank_statistics.csv             # Từ Google Drive
├── model_comparison.csv            # Từ Google Drive
├── best_model.pkl                  # Từ Google Drive
├── tfidf_vectorizer.pkl            # Từ Google Drive
└── model_metadata.json             # Từ Google Drive
```

#### Chạy Dashboard

**Phương án A: Script tự động (khuyến nghị)**

Windows:

```cmd
run_dashboard.bat
```

macOS/Linux:

```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

Script sẽ tự động:

- ✅ Tạo virtual environment
- ✅ Cài đặt thư viện
- ✅ Kiểm tra file dữ liệu
- ✅ Chạy Streamlit

**Phương án B: Chạy thủ công**

```bash
# 1. Tạo virtual environment (khuyến nghị)
python3 -m venv venv

# 2. Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Chạy dashboard
streamlit run dashboard.py
```

#### Truy cập Dashboard

Mở browser tại: **http://localhost:8501**

Nếu port 8501 bị chiếm:

```bash
streamlit run dashboard.py --server.port 8502
```

---

## 🎨 Sử dụng Dashboard

### Tab 1: 📊 Tổng quan

- Xem metrics tổng thể
- Biểu đồ phân bố sentiment/rating
- Xu hướng theo thời gian

### Tab 2: 🏦 Phân tích theo ngân hàng

- Rating trung bình từng ngân hàng
- Tỷ lệ reviews tích cực
- Bảng thống kê chi tiết

### Tab 3: 🤖 So sánh mô hình ML

- Hiệu suất 3 models
- Thời gian training/prediction
- Metrics chi tiết

### Tab 4: 🔮 Dự đoán Sentiment

- **Nhập review mới** → AI dự đoán cảm xúc
- Xem độ tin cậy
- Test với các ví dụ

### Tab 5: 📝 Dữ liệu chi tiết

- Xem toàn bộ reviews
- Filter theo ngân hàng/sentiment/rating
- Download dữ liệu

### Sidebar: Bộ lọc

- Chọn ngân hàng
- Chọn sentiment
- Chọn khoảng rating

---

## 🔧 Xử lý lỗi

### Lỗi 1: File không tồn tại

```
FileNotFoundError: reviews_with_sentiment.csv
```

**Giải pháp:**

- Kiểm tra đã download đủ 6 files từ Google Drive
- Đảm bảo files ở cùng thư mục với `dashboard.py`

### Lỗi 2: Module not found

```
ModuleNotFoundError: No module named 'streamlit'
```

**Giải pháp:**

```bash
pip install streamlit
# hoặc
pip install -r requirements.txt
```

### Lỗi 3: Port đã được sử dụng

```
Address already in use
```

**Giải pháp:**

```bash
# Đổi port
streamlit run dashboard.py --server.port 8502
```

### Lỗi 4: Underthesea không chạy

```
Error in word_tokenize
```

**Giải pháp:**

```bash
pip install --upgrade underthesea
```

### Lỗi 5: Crawl bị chặn (Google Colab)

```
Error crawling: Too many requests
```

**Giải pháp:**

- Chờ 5-10 phút rồi chạy lại
- Giảm `count` trong function `reviews()` xuống 50-100

---

## 📈 Kết quả mong đợi

### Dataset

- **10,000 - 15,000** reviews
- **12** ngân hàng
- **2** nhãn (tích cực/tiêu cực)

### Models Performance

- Accuracy: **85-88%**
- F1-Score: **85-88%**
- Best Model: **SVM Linear**

### Dashboard Features

- ✅ 5 tabs tương tác
- ✅ Biểu đồ động (Plotly)
- ✅ Dự đoán real-time
- ✅ Filter & search
- ✅ Export data

---

## ⏱️ Timeline

| Bước     | Mô tả           | Thời gian       |
| -------- | --------------- | --------------- |
| 1        | Setup Colab     | 2 phút          |
| 2        | Crawl data      | 15-20 phút      |
| 3        | Clean & label   | 2-3 phút        |
| 4        | Training models | 5-10 phút       |
| 5        | Visualization   | 1-2 phút        |
| 6        | Download files  | 1 phút          |
| 7        | Setup local     | 2 phút          |
| 8        | Run dashboard   | 1 phút          |
| **TỔNG** |                 | **~30-45 phút** |

---

## 💡 Tips

### Tối ưu tốc độ Crawl

```python
# Giảm số lượng reviews mỗi query
count=50  # thay vì 200
```

### Tăng accuracy

```python
# Tăng max_features của TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,  # tăng từ 5000
    ngram_range=(1, 3)   # thêm trigram
)
```

### Dashboard chạy nhanh hơn

```python
# Giảm số lượng rows hiển thị
st.dataframe(df.head(1000))  # thay vì toàn bộ
```

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:

1. **Đọc README.md** - Hướng dẫn chi tiết
2. **Xem phần Xử lý lỗi** ở trên
3. **Check file logs** - Streamlit tự động log lỗi
4. **Google error message** - Nhiều người gặp vấn đề tương tự

---

## ✅ Checklist hoàn thành

- [ ] Đã chạy kltn.py trên Colab thành công
- [ ] Đã có đủ 6 files quan trọng
- [ ] Đã cài đặt Python 3.8+
- [ ] Đã cài đặt pip
- [ ] Đã cài đặt requirements.txt
- [ ] Dashboard chạy được tại localhost:8501
- [ ] Có thể dự đoán sentiment cho review mới
- [ ] Có thể xem được tất cả 5 tabs
- [ ] Có thể export dữ liệu

---

## 🎉 Hoàn thành!

Chúc mừng! Bạn đã hoàn thành:

- ✅ Thu thập 10k+ reviews
- ✅ Xây dựng 3 ML models
- ✅ Tạo web dashboard chuyên nghiệp
- ✅ Phân tích sentiment 12 ngân hàng

**→ Đã sẵn sàng demo đồ án tốt nghiệp! 🎓**

---

## 📚 Tài liệu thêm

- [README.md](README.md) - Hướng dẫn đầy đủ
- [YEU_CAU.md](YEU_CAU.md) - Yêu cầu đồ án
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Underthesea Docs](https://underthesea.readthedocs.io/)
