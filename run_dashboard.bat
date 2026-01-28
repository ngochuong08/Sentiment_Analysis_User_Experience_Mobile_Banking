@echo off
REM Script chạy nhanh cho dự án Sentiment Analysis trên Windows

echo ==================================
echo 🏦 SENTIMENT ANALYSIS DASHBOARD
echo ==================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python chưa được cài đặt
    echo 💡 Vui lòng cài đặt Python 3.8 trở lên
    pause
    exit /b 1
)

echo ✅ Python đã được cài đặt
python --version
echo.

REM Kiểm tra pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip chưa được cài đặt
    pause
    exit /b 1
)

echo ✅ pip đã được cài đặt
echo.

REM Kiểm tra virtual environment
if not exist "venv" (
    echo 📦 Tạo virtual environment...
    python -m venv venv
    echo ✅ Đã tạo virtual environment
    echo.
)

REM Activate virtual environment
echo 🔧 Kích hoạt virtual environment...
call venv\Scripts\activate.bat

REM Kiểm tra đã cài đặt thư viện chưa
if not exist "venv\.installed" (
    echo 📥 Cài đặt các thư viện cần thiết...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    REM Đánh dấu đã cài đặt
    echo. > venv\.installed
    echo ✅ Đã cài đặt tất cả thư viện
    echo.
) else (
    echo ✅ Các thư viện đã được cài đặt trước đó
    echo.
)

REM Kiểm tra file dữ liệu
echo 🔍 Kiểm tra dữ liệu...
if not exist "reviews_with_sentiment.csv" (
    echo ⚠️  Chưa có file reviews_with_sentiment.csv
    echo 💡 Hướng dẫn:
    echo    1. Chạy file kltn.py trên Google Colab
    echo    2. Download các file CSV từ Google Drive
    echo    3. Đặt các file vào thư mục này
    echo.
    echo 📋 Các file cần thiết:
    echo    - reviews_with_sentiment.csv
    echo    - bank_statistics.csv
    echo    - model_comparison.csv
    echo    - best_model.pkl
    echo    - tfidf_vectorizer.pkl
    echo    - model_metadata.json
    echo.
    
    set /p continue="❓ Bạn đã có đủ file chưa? (y/n): "
    if /i not "%continue%"=="y" (
        echo 👋 Vui lòng chuẩn bị dữ liệu trước khi chạy dashboard
        pause
        exit /b 0
    )
)

echo ✅ Đã tìm thấy file dữ liệu
echo.

REM Chạy dashboard
echo ==================================
echo 🚀 KHỞI ĐỘNG DASHBOARD
echo ==================================
echo.
echo 📊 Dashboard sẽ mở tại: http://localhost:8501
echo 🛑 Nhấn Ctrl+C để dừng
echo.

streamlit run dashboard.py

REM Deactivate khi thoát
call venv\Scripts\deactivate.bat
