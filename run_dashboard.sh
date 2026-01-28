#!/bin/bash
# Script chạy nhanh cho dự án Sentiment Analysis

echo "=================================="
echo "🏦 SENTIMENT ANALYSIS DASHBOARD"
echo "=================================="
echo ""

# Kiểm tra Python
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 chưa được cài đặt"
    echo "💡 Vui lòng cài đặt Python 3.8 trở lên"
    exit 1
fi

echo "✅ Python version:"
python3 --version
echo ""

# Kiểm tra pip
if ! command -v pip3 &> /dev/null
then
    echo "❌ pip3 chưa được cài đặt"
    exit 1
fi

echo "✅ pip3 đã được cài đặt"
echo ""

# Kiểm tra xem đã có virtual environment chưa
if [ ! -d "venv" ]; then
    echo "📦 Tạo virtual environment..."
    python3 -m venv venv
    echo "✅ Đã tạo virtual environment"
    echo ""
fi

# Activate virtual environment
echo "🔧 Kích hoạt virtual environment..."
source venv/bin/activate

# Kiểm tra xem đã cài đặt thư viện chưa
if [ ! -f "venv/.installed" ]; then
    echo "📥 Cài đặt các thư viện cần thiết..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Đánh dấu đã cài đặt
    touch venv/.installed
    echo "✅ Đã cài đặt tất cả thư viện"
    echo ""
else
    echo "✅ Các thư viện đã được cài đặt trước đó"
    echo ""
fi

# Kiểm tra file dữ liệu
echo "🔍 Kiểm tra dữ liệu..."
if [ ! -f "reviews_with_sentiment.csv" ]; then
    echo "⚠️  Chưa có file reviews_with_sentiment.csv"
    echo "💡 Hướng dẫn:"
    echo "   1. Chạy file kltn.py trên Google Colab"
    echo "   2. Download các file CSV từ Google Drive"
    echo "   3. Đặt các file vào thư mục này"
    echo ""
    echo "📋 Các file cần thiết:"
    echo "   - reviews_with_sentiment.csv"
    echo "   - bank_statistics.csv"
    echo "   - model_comparison.csv"
    echo "   - best_model.pkl"
    echo "   - tfidf_vectorizer.pkl"
    echo "   - model_metadata.json"
    echo ""
    
    read -p "❓ Bạn đã có đủ file chưa? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "👋 Vui lòng chuẩn bị dữ liệu trước khi chạy dashboard"
        exit 0
    fi
fi

echo "✅ Đã tìm thấy file dữ liệu"
echo ""

# Chạy dashboard
echo "=================================="
echo "🚀 KHỞI ĐỘNG DASHBOARD"
echo "=================================="
echo ""
echo "📊 Dashboard sẽ mở tại: http://localhost:8501"
echo "🛑 Nhấn Ctrl+C để dừng"
echo ""

streamlit run dashboard.py

# Deactivate khi thoát
deactivate
