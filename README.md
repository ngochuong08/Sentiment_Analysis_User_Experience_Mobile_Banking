# 🏦 Sentiment Analysis - Mobile Banking Applications

Phân tích cảm xúc người dùng đối với các ứng dụng ngân hàng di động tại Việt Nam sử dụng Machine Learning và Natural Language Processing.

---

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Kết quả](#kết-quả)
- [Demo](#demo)

---

## 🎯 Giới thiệu

Dự án này thực hiện phân tích cảm xúc (sentiment analysis) trên các đánh giá của người dùng về 12 ứng dụng ngân hàng hàng đầu Việt Nam trên Google Play Store, bao gồm:

- **BIDV** - Ngân hàng TMCP Đầu tư và Phát triển Việt Nam
- **Agribank** - Ngân hàng Nông nghiệp và Phát triển Nông thôn
- **Vietcombank** - Ngân hàng TMCP Ngoại thương Việt Nam
- **Sacombank** - Ngân hàng TMCP Sài Gòn Thương Tín
- **MBBank** - Ngân hàng TMCP Quân Đội
- **VietinBank** - Ngân hàng TMCP Công Thương Việt Nam
- **TPBank** - Ngân hàng TMCP Tiên Phong
- **Techcombank** - Ngân hàng TMCP Kỹ Thương Việt Nam
- **HDBank** - Ngân hàng TMCP Phát triển TP.HCM
- **ACB** - Ngân hàng TMCP Á Châu
- **OCB** - Ngân hàng TMCP Phương Đông
- **MSB** - Ngân hàng TMCP Hàng Hải

### Mục tiêu

1. Thu thập và xử lý dữ liệu reviews từ Google Play Store
2. Tiền xử lý dữ liệu tiếng Việt (xử lý teencode, emoji, lỗi chính tả)
3. Gán nhãn sentiment dựa trên rating
4. Xây dựng và so sánh các mô hình Machine Learning
5. Xây dựng web dashboard trực quan để phân tích

---

## ✨ Tính năng

### 1. Thu thập & Xử lý dữ liệu

- ✅ Crawl reviews từ Google Play Store
- ✅ Xử lý teencode tiếng Việt
- ✅ Loại bỏ emoji và ký tự đặc biệt
- ✅ Sửa lỗi chính tả phổ biến
- ✅ Tách từ tiếng Việt (word segmentation)

### 2. Machine Learning Models

- ✅ **Naive Bayes** - Mô hình xác suất đơn giản, hiệu quả
- ✅ **Logistic Regression** - Mô hình tuyến tính với khả năng giải thích tốt
- ✅ **SVM (Linear)** - Support Vector Machine cho phân loại văn bản

### 3. Phân tích & Visualization

- ✅ So sánh hiệu suất các mô hình (Accuracy, Precision, Recall, F1-Score)
- ✅ Phân tích sentiment theo từng ngân hàng
- ✅ Biểu đồ phân bố rating và sentiment
- ✅ Word Cloud cho reviews tích cực/tiêu cực
- ✅ Xu hướng sentiment theo thời gian

### 4. Web Dashboard

- ✅ Giao diện tương tác với Streamlit
- ✅ Dự đoán sentiment cho review mới
- ✅ Lọc và tìm kiếm dữ liệu
- ✅ Visualization động với Plotly
- ✅ Export dữ liệu CSV

---

## 🛠 Công nghệ sử dụng

### Backend & ML

- **Python 3.8+**
- **pandas** - Xử lý và phân tích dữ liệu
- **numpy** - Tính toán số học
- **scikit-learn** - Machine Learning algorithms
- **underthesea** - Xử lý ngôn ngữ tự nhiên tiếng Việt
- **google-play-scraper** - Thu thập dữ liệu từ Google Play

### Visualization

- **matplotlib** - Vẽ biểu đồ cơ bản
- **plotly** - Biểu đồ tương tác
- **wordcloud** - Tạo word cloud

### Web Dashboard

- **Streamlit** - Framework xây dựng web app nhanh chóng

---

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- pip (Python package manager)
- 4GB RAM trở lên
- Google Account (để chạy trên Colab hoặc lưu vào Google Drive)

### Cài đặt trên máy local

1. **Clone repository hoặc tải code**

```bash
cd /path/to/your/project
```

2. **Tạo virtual environment (khuyến nghị)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Cài đặt thư viện**

```bash
pip install pandas numpy matplotlib seaborn plotly
pip install scikit-learn imbalanced-learn
pip install google-play-scraper
pip install underthesea
pip install wordcloud
pip install streamlit
pip install tqdm
```

Hoặc tạo file `requirements.txt`:

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
google-play-scraper>=1.2.0
underthesea>=1.3.0
wordcloud>=1.8.2
streamlit>=1.28.0
tqdm>=4.64.0
Pillow<12
```

Sau đó cài đặt:

```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn sử dụng

### Phương án 1: Chạy trên Google Colab (Khuyến nghị)

#### Bước 1: Crawl và Xử lý dữ liệu

1. Mở Google Colab: https://colab.research.google.com/
2. Upload file `kltn.py` hoặc copy nội dung vào notebook mới
3. Chạy từng cell theo thứ tự:

```python
# Cell 1: Cài đặt thư viện
!pip install -qq google-play-scraper
!pip install -qq -U watermark tqdm
!pip install -q "pillow<12"
!pip install -qq underthesea
!pip install -qq scikit-learn
!pip install -qq imbalanced-learn
!pip install -qq wordcloud
```

```python
# Cell 2: Import thư viện
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# ... (các import khác)
```

```python
# Cell 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

4. Chạy tiếp các cell còn lại để:
   - Thu thập dữ liệu reviews (mất khoảng 15-30 phút)
   - Làm sạch dữ liệu
   - Gán nhãn sentiment
   - Training các mô hình ML
   - Tạo visualization

5. Kết quả sẽ được lưu vào Google Drive tại:
   - `/content/drive/MyDrive/google_play_scraper/`

#### Bước 2: Chạy Web Dashboard

**Chạy trên Colab (cách 1):**

```python
!pip install streamlit

# Tạo tunnel để truy cập từ browser
!npm install -g localtunnel

# Chạy streamlit trong background
!streamlit run dashboard.py &

# Tạo public URL
!npx localtunnel --port 8501
```

**Chạy trên máy local (cách 2 - khuyến nghị):**

1. Download các file từ Google Drive:
   - `reviews_with_sentiment.csv`
   - `bank_statistics.csv`
   - `model_comparison.csv`
   - `best_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `model_metadata.json`

2. Đặt các file vào cùng thư mục với `dashboard.py`
3. Chạy lệnh:

```bash
streamlit run dashboard.py
```

4. Trình duyệt sẽ tự động mở tại: `http://localhost:8501`

---

### Phương án 2: Chạy hoàn toàn trên máy local

#### Bước 1: Chuyển đổi file kltn.py

Do file `kltn.py` được thiết kế cho Google Colab (có các lệnh `!pip` và `%watermark`), bạn cần:

**Cách 1: Chạy trực tiếp**

```bash
# Bỏ qua các lỗi với !pip và %watermark
python kltn.py 2>/dev/null
```

**Cách 2: Tạo script mới**

Tạo file `run_local.py` và copy nội dung từ `kltn.py`, loại bỏ:

- Các dòng bắt đầu bằng `!pip`
- Các dòng bắt đầu bằng `%`
- Code mount Google Drive

Thay đổi đường dẫn:

```python
# Thay vì
SAVE_PATH = "/content/drive/MyDrive/google_play_scraper"

# Dùng
SAVE_PATH = "./data"
```

#### Bước 2: Chạy script

```bash
python run_local.py
```

#### Bước 3: Chạy Dashboard

```bash
streamlit run dashboard.py
```

---

## 📁 Cấu trúc dự án

```
Sentiment_Analysis_Mobile_Banking/
│
├── kltn.py                          # Script chính (Google Colab)
├── dashboard.py                     # Web dashboard (Streamlit)
├── YEU_CAU.md                       # Yêu cầu đồ án
├── README.md                        # Tài liệu hướng dẫn
├── requirements.txt                 # Danh sách thư viện
│
├── data/                            # Thư mục dữ liệu
│   ├── apps_info.csv               # Thông tin ứng dụng
│   ├── apps_reviews.csv            # Reviews gốc
│   ├── reviews_cleaned.csv         # Reviews đã làm sạch
│   ├── reviews_with_sentiment.csv  # Reviews có nhãn
│   ├── reviews_for_ml.csv          # Dữ liệu cho ML
│   ├── reviews_negative.csv        # Reviews tiêu cực
│   ├── reviews_positive.csv        # Reviews tích cực
│   ├── bank_statistics.csv         # Thống kê theo ngân hàng
│   └── model_comparison.csv        # So sánh mô hình
│
├── models/                          # Thư mục models
│   ├── best_model.pkl              # Mô hình tốt nhất
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   └── model_metadata.json         # Metadata mô hình
│
└── visualizations/                  # Thư mục biểu đồ
    ├── sentiment_distribution.png
    ├── model_comparison.png
    ├── bank_comparison.png
    └── wordcloud.png
```

---

## 📊 Kết quả

### Dataset

- **Tổng số reviews**: ~10,000 - 15,000 reviews
- **Số ngân hàng**: 12 ngân hàng
- **Khoảng thời gian**: Dữ liệu gần nhất từ Google Play

### Hiệu suất mô hình (Ví dụ)

| Model               | Accuracy | Precision | Recall | F1-Score | Training Time |
| ------------------- | -------- | --------- | ------ | -------- | ------------- |
| Naive Bayes         | 0.8523   | 0.8542    | 0.8523 | 0.8498   | 0.15s         |
| Logistic Regression | 0.8742   | 0.8756    | 0.8742 | 0.8738   | 0.89s         |
| SVM (Linear)        | 0.8798   | 0.8812    | 0.8798 | 0.8794   | 1.23s         |

**Mô hình tốt nhất**: SVM (Linear) với F1-Score ~0.88

### Phân tích Sentiment

- **Reviews tích cực (4-5 sao)**: ~60-70%
- **Reviews tiêu cực (1-3 sao)**: ~30-40%

### Top Banking Apps

1. **Techcombank** - Rating TB: 4.2⭐
2. **MBBank** - Rating TB: 4.1⭐
3. **TPBank** - Rating TB: 4.0⭐

---

## 🎨 Demo

### Web Dashboard Features

#### 1. Tổng quan

![Dashboard Overview](https://via.placeholder.com/800x400?text=Dashboard+Overview)

- Metrics tổng quan
- Biểu đồ phân bố sentiment
- Xu hướng theo thời gian

#### 2. Phân tích theo ngân hàng

![Bank Analysis](https://via.placeholder.com/800x400?text=Bank+Analysis)

- Rating trung bình
- Tỷ lệ reviews tích cực
- Bảng thống kê chi tiết

#### 3. So sánh mô hình

![Model Comparison](https://via.placeholder.com/800x400?text=Model+Comparison)

- Hiệu suất các mô hình
- Thời gian training/prediction
- Metrics chi tiết

#### 4. Dự đoán Sentiment

![Prediction](https://via.placeholder.com/800x400?text=Prediction+Feature)

- Nhập review mới
- Dự đoán real-time
- Độ tin cậy

---

## 🐛 Xử lý lỗi thường gặp

### Lỗi 1: ModuleNotFoundError

```bash
# Lỗi
ModuleNotFoundError: No module named 'underthesea'

# Giải pháp
pip install underthesea
```

### Lỗi 2: File not found (Dashboard)

```bash
# Lỗi
FileNotFoundError: reviews_with_sentiment.csv

# Giải pháp
# Đảm bảo các file CSV ở cùng thư mục với dashboard.py
# Hoặc sửa đường dẫn trong dashboard.py:
df = pd.read_csv('./data/reviews_with_sentiment.csv')
```

### Lỗi 3: Streamlit port đã được sử dụng

```bash
# Lỗi
Address already in use

# Giải pháp
streamlit run dashboard.py --server.port 8502
```

### Lỗi 4: Crawl bị chặn

```python
# Lỗi
Error crawling: Too many requests

# Giải pháp
# Thêm delay giữa các requests
import time
time.sleep(2)  # Chờ 2 giây giữa mỗi request
```

### Lỗi 5: Hết bộ nhớ khi training

```python
# Giải pháp
# Giảm max_features của TF-IDF
tfidf = TfidfVectorizer(
    max_features=3000,  # Giảm từ 5000 xuống 3000
    min_df=3,
    max_df=0.7
)
```

---

## 📝 Ghi chú

### Giới hạn của Google Play Scraper

- Số lượng reviews tối đa mỗi lần crawl: ~200 reviews/query
- Có thể bị rate limit nếu request quá nhiều
- Dữ liệu có thể thay đổi theo thời gian

### Cải tiến trong tương lai

- [ ] Sử dụng mô hình Deep Learning (LSTM, BERT)
- [ ] Thêm phân tích chủ đề (Topic Modeling)
- [ ] Crawl từ App Store (iOS)
- [ ] Phát hiện spam/fake reviews
- [ ] API để tích hợp vào hệ thống khác
- [ ] Real-time monitoring

---

## 🤝 Đóng góp

Dự án này là đồ án tốt nghiệp, nhưng rất hoan nghênh mọi đóng góp:

1. Fork dự án
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

# 📧 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ qua email hoặc tạo issue trên GitHub.

---

## 🙏 Cảm ơn

- **Underthesea** - Thư viện xử lý NLP tiếng Việt tuyệt vời
- **Streamlit** - Framework web app đơn giản và mạnh mẽ
- **Google Play Scraper** - Công cụ thu thập dữ liệu hiệu quả

---

**⭐ Nếu dự án hữu ích, đừng quên star repo nhé! ⭐**
