# 🚀 HƯỚNG DẪN CHẠY TRÊN KAGGLE

## 📋 Giới thiệu

File `kltn_kaggle.py` đã được tối ưu hóa để chạy trên **Kaggle Notebooks** với các cải tiến:

✅ **Loại bỏ lỗi**:

- Không cần mount Google Drive
- Tự động cài đặt thư viện thiếu
- Xử lý exceptions tốt hơn
- Tối ưu cho môi trường Kaggle

✅ **Tính năng mới**:

- Tự động loại bỏ reviews trùng lặp
- Rate limiting để tránh bị chặn
- Kiểm tra và hiển thị tiến trình rõ ràng
- Tự động tạo thư mục output

---

## 🎯 CÁCH CHẠY NHANH (3 PHÚT)

### Bước 1: Upload lên Kaggle

1. Truy cập: https://www.kaggle.com/
2. Đăng nhập tài khoản
3. Click **"New Notebook"** → **"Create"**
4. Click **"File"** → **"Upload Notebook"**
5. Upload file `kltn_kaggle.py`

### Bước 2: Cấu hình Notebook

1. **Settings** (⚙️ bên phải):
   - **Language**: Python
   - **Environment**: Latest (Python 3.10+)
   - **Accelerator**: None (CPU đủ)
   - **Persistence**: Files only
   - **Internet**: ON ✅ (BẮT BUỘC để crawl data)

2. **Save Version**:
   - Click **"Save Version"**
   - Chọn **"Save & Run All"**

### Bước 3: Chờ hoàn thành

- ⏱️ Thời gian chạy: **30-45 phút**
- 📊 Tiến trình hiển thị real-time
- ✅ Tự động lưu kết quả

### Bước 4: Download kết quả

1. Click tab **"Output"** bên phải
2. Click **"View All"**
3. Download folder **"output"** chứa:
   - ✅ 10+ CSV files
   - ✅ 4 PNG biểu đồ
   - ✅ 2 PKL models
   - ✅ 1 JSON metadata

---

## 📂 CẤU TRÚC OUTPUT

Tất cả files được lưu trong `/kaggle/working/output/`:

```
output/
├── apps_info.csv                    # Thông tin 12 ngân hàng
├── apps_reviews.csv                 # ~10k-15k reviews gốc
├── reviews_cleaned.csv              # Reviews đã clean
├── reviews_with_sentiment.csv       # ⭐ FILE QUAN TRỌNG
├── reviews_for_ml.csv               # Data cho ML
├── bank_statistics.csv              # ⭐ Thống kê ngân hàng
├── model_comparison.csv             # ⭐ So sánh models
├── best_model.pkl                   # ⭐ Model tốt nhất
├── tfidf_vectorizer.pkl             # ⭐ Vectorizer
├── model_metadata.json              # ⭐ Metadata
├── sentiment_distribution.png       # Biểu đồ sentiment
├── model_comparison.png             # Biểu đồ so sánh models
├── bank_comparison.png              # Biểu đồ ngân hàng
└── wordcloud.png                    # Word cloud
```

---

## 🔧 KHÁC BIỆT VỚI COLAB VERSION

| Feature     | Google Colab         | Kaggle                |
| ----------- | -------------------- | --------------------- |
| Mount Drive | ✅ Cần               | ❌ Không cần          |
| Internet    | ✅ Có sẵn            | ⚠️ Phải bật           |
| Thư viện    | Cài thủ công         | Auto detect           |
| Output Path | `/content/drive/...` | `/kaggle/working/...` |
| Download    | Từ Drive             | Từ Output tab         |
| Rate Limit  | Ít nghiêm            | Cần delay             |
| Runtime     | 12 giờ               | 9 giờ                 |

---

## ⚙️ CÀI ĐẶT CHI TIẾT

### 1. Bật Internet (BẮT BUỘC)

Code cần Internet để:

- Crawl reviews từ Google Play
- Cài đặt thư viện thiếu
- Download models

**Cách bật**:

```
Settings → Internet → ON
```

### 2. Cấu hình GPU/TPU (KHÔNG CẦN)

Code chỉ dùng CPU, không cần GPU/TPU:

```
Settings → Accelerator → None
```

### 3. Thư viện tự động cài

Code tự động cài:

- `google-play-scraper`
- `underthesea`
- `wordcloud`
- `imbalanced-learn`

Các thư viện khác có sẵn trên Kaggle.

---

## 📊 TIẾN TRÌNH THỰC THI

### Phase 1: Setup (2 phút)

```
✅ Cài đặt thư viện
✅ Import libraries
✅ Tạo output directory
```

### Phase 2: Crawl Data (15-25 phút)

```
📥 Thu thập thông tin 12 app
📥 Crawl ~10k-15k reviews
   - Score 1-5 sao
   - Sort: Most Relevant + Newest
   - Delay 1s giữa mỗi request
```

### Phase 3: Data Processing (3-5 phút)

```
🧹 Làm sạch dữ liệu
   - Loại emoji, teencode
   - Chuẩn hóa text
   - Tách từ tiếng Việt
🏷️ Gán nhãn sentiment
📊 Tạo visualization
```

### Phase 4: Machine Learning (5-10 phút)

```
🤖 Training 3 models:
   - Naive Bayes
   - Logistic Regression
   - SVM (Linear)
📈 So sánh hiệu suất
💾 Lưu model tốt nhất
```

### Phase 5: Analysis (2-3 phút)

```
🏦 Phân tích theo ngân hàng
📊 Tạo biểu đồ
💬 Word cloud
💾 Lưu tất cả files
```

**TỔNG: 30-45 phút**

---

## 🐛 XỬ LÝ LỖI

### Lỗi 1: Internet disabled

```
Error: Connection refused
```

**Giải pháp**:

```
Settings → Internet → ON ✅
Save & Run All
```

### Lỗi 2: Rate limit

```
Error: Too many requests
```

**Giải pháp**:

- Code đã có `time.sleep(1)` tự động
- Nếu vẫn lỗi, tăng delay:

```python
time.sleep(2)  # Thay vì 1
```

### Lỗi 3: Module not found

```
ModuleNotFoundError: No module named 'underthesea'
```

**Giải pháp**:

- Code sẽ tự động cài
- Nếu lỗi, chạy thủ công:

```python
!pip install underthesea
```

### Lỗi 4: Timeout khi crawl

```
TimeoutError
```

**Giải pháp**:

- Giảm `count` từ 200 → 100
- Hoặc bỏ qua app bị lỗi (code đã xử lý)

### Lỗi 5: Memory limit

```
MemoryError
```

**Giải pháp**:

- Giảm `max_features` từ 5000 → 3000

```python
tfidf = TfidfVectorizer(max_features=3000)
```

---

## 💡 TIPS & TRICKS

### 1. Tăng tốc độ crawl

```python
# Giảm số lượng reviews
count = 50  # thay vì 100/200
```

### 2. Test nhanh với 2 ngân hàng

```python
APP_PACKAGES = [
    'com.vnpay.bidv',
    'com.mbmobile',
]
```

### 3. Tối ưu accuracy

```python
# Tăng features
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3)  # thêm trigram
)
```

### 4. Giảm thời gian chạy

```python
# Bỏ qua word cloud (tốn thời gian)
# Comment dòng tạo word cloud
```

### 5. Check tiến trình

```python
# Xem output ngay lập tức
print("Current step:", step_name)
```

---

## 📥 DOWNLOAD KẾT QUẢ

### Cách 1: Download từ UI

1. Click tab **"Output"** (bên phải)
2. Hover vào file/folder
3. Click icon **Download** (⬇️)

### Cách 2: Download tất cả

1. Click **"View All"** trong Output tab
2. Select folder **"output"**
3. Click **"Download"**
4. Giải nén file ZIP

### Cách 3: Code download

```python
# Nén tất cả files
import shutil
shutil.make_archive('/kaggle/working/all_results', 'zip', OUTPUT_DIR)
```

---

## 🎓 SỬ DỤNG KẾT QUẢ

### 1. Chạy Dashboard local

Download 6 files:

- `reviews_with_sentiment.csv`
- `bank_statistics.csv`
- `model_comparison.csv`
- `best_model.pkl`
- `tfidf_vectorizer.pkl`
- `model_metadata.json`

Đặt vào cùng folder với `dashboard.py`, chạy:

```bash
streamlit run dashboard.py
```

### 2. Tiếp tục phân tích

Upload files vào notebook mới:

```python
import pandas as pd
df = pd.read_csv('/kaggle/input/your-dataset/reviews_with_sentiment.csv')
```

### 3. Load model để predict

```python
import pickle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
```

---

## 📊 KẾT QUẢ MONG ĐỢI

### Dataset

- **10,000-15,000** reviews
- **12** ngân hàng
- **2** nhãn (tích cực/tiêu cực)

### Models Performance

| Model               | Accuracy | F1-Score |
| ------------------- | -------- | -------- |
| Naive Bayes         | ~85%     | ~85%     |
| Logistic Regression | ~87%     | ~87%     |
| **SVM (Linear)**    | **~88%** | **~88%** |

### Phân tích

- ✅ So sánh 12 ngân hàng
- ✅ Word cloud tích cực/tiêu cực
- ✅ Phân bố sentiment/rating
- ✅ Classification report chi tiết

---

## 🆚 SO SÁNH PLATFORMS

### Kaggle Pros

✅ Không cần Google account
✅ Dễ share notebook (public)
✅ Download files dễ dàng
✅ Community support tốt
✅ Free tier đủ dùng

### Kaggle Cons

⚠️ Runtime 9 giờ (ít hơn Colab)
⚠️ Phải bật Internet manually
⚠️ Cần cẩn thận rate limit
⚠️ Ít flexible hơn

### Colab Pros

✅ Runtime 12 giờ
✅ Internet mặc định
✅ Tích hợp Drive tốt
✅ GPU/TPU free

### Colab Cons

⚠️ Cần Google account
⚠️ Drive mount phức tạp
⚠️ Khó share results

---

## ✅ CHECKLIST HOÀN THÀNH

- [ ] Đã tạo Kaggle account
- [ ] Đã upload notebook
- [ ] Đã bật Internet
- [ ] Đã chạy "Save & Run All"
- [ ] Code chạy không lỗi
- [ ] Có 14+ files trong output
- [ ] Đã download tất cả files
- [ ] Models có accuracy >85%
- [ ] Có thể load model để predict
- [ ] Ready để demo!

---

## 🎉 HOÀN THÀNH!

Bây giờ bạn đã có:

- ✅ Dataset 10k+ reviews
- ✅ 3 ML models trained
- ✅ Phân tích chi tiết 12 ngân hàng
- ✅ Visualization chuyên nghiệp
- ✅ Files sẵn sàng cho dashboard

**→ Sẵn sàng demo đồ án! 🎓**

---

## 📚 TÀI LIỆU THÊM

- **Kaggle Docs**: https://www.kaggle.com/docs
- **API Reference**: https://www.kaggle.com/docs/api
- **Community**: https://www.kaggle.com/discussions

---

## 📧 HỖ TRỢ

Nếu gặp vấn đề:

1. Check phần "Xử lý lỗi" ở trên
2. Xem log trong notebook
3. Google error message
4. Hỏi Kaggle community

---

**✨ Chúc bạn thành công với đồ án! 🚀**
