TÓM TẮT FILE IMPROVED
CÁC CẢI TIẾN CHÍNH (so với kltn_kaggle.py):

0. Data Crawling Improvements

   Lấy 500 reviews thay ví 100-200

   Thêm exponential backoff retry

   Dynamic rate limiting

   Progress tracking chi tiết

1. Text Cleaning (150+ teencode)
   Base: 50 teencode words
   Improved: 150+ teencode words
   Unicode normalization (NFC)
   Vietnamese stopwords removal
   Advanced typo correction
   Negation handling (xử lý phủ định)
   Phủ định + Từ tiêu cực = Tích cực
   Phủ định + Từ tích cực = Tiêu cực
2. Feature Engineering
   Base: TF-IDF bigrams (1,2) - 5,000 features
   Improved: TF-IDF trigrams (1,3) - 5,000 features

   Char-level TF-IDF (2,4) - 2,000 features
   = Combined 7,000+ features

3. Models
   Base: 3 models (NB, LR, SVM)
   Improved: 5 models (+ Random Forest, XGBoost)

   Ensemble Voting Classifier

4. Hyperparameter Tuning
   Base: Default parameters
   Improved: GridSearchCV (3-12 combinations per model)
   5-fold cross-validation
5. Imbalanced Data
   Base: Không xử lý
   Improved: SMOTE oversampling + class_weight='balanced'
6. Data Crawling
   Base: Fixed retry, simple error handling
   Improved: Exponential backoff, advanced retry logic
7. Evaluation
   Base: Basic metrics (Acc, Prec, Rec, F1)
   Improved:
   ROC-AUC curves (all models)
   Feature importance (RF, XGBoost)
   Error analysis (misclassified examples)
   Confusion matrix heatmaps

- F1-Score \~87-88% là **rất tốt** cho traditional ML
- Nếu cần >90%, phải dùng Deep Learning (BERT, PhoBERT)
