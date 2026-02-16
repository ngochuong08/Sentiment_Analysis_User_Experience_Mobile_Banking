# -*- coding: utf-8 -*-
"""
STOCK DATA CRAWLER & FINANCIAL CORRELATION ANALYSIS
=====================================================
Crawl giá cổ phiếu 11 ngân hàng niêm yết trên HOSE
Merge với sentiment data theo tuần
Tính tương quan (Pearson, Spearman) và Granger Causality

Author: KLTN - Sentiment Analysis Mobile Banking
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
import time

warnings.filterwarnings("ignore")

# ========================================
# CONFIG
# ========================================
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping: appId → bank name → stock ticker
BANK_STOCK_MAP = {
    "com.vnpay.bidv": {"bank": "BIDV", "ticker": "BID.VN"},
    "com.VCB": {"bank": "Vietcombank", "ticker": "VCB.VN"},
    "com.vietinbank.ipay": {"bank": "VietinBank", "ticker": "CTG.VN"},
    "com.mbmobile": {"bank": "MBBank", "ticker": "MBB.VN"},
    "vn.com.techcombank.bb.app": {"bank": "Techcombank", "ticker": "TCB.VN"},
    "com.sacombank.ewallet": {"bank": "Sacombank", "ticker": "STB.VN"},
    "com.tpb.mb.gprsandroid": {"bank": "TPBank", "ticker": "TPB.VN"},
    "com.vnpay.hdbank": {"bank": "HDBank", "ticker": "HDB.VN"},
    "mobile.acb.com.vn": {"bank": "ACB", "ticker": "ACB.VN"},
    "vn.com.ocb.awe": {"bank": "OCB", "ticker": "OCB.VN"},
    "vn.com.msb.smartBanking": {"bank": "MSB", "ticker": "MSB.VN"},
    # Agribank: KHÔNG niêm yết → loại khỏi phân tích tài chính
}

BANK_NAME_TO_TICKER = {v["bank"]: v["ticker"] for v in BANK_STOCK_MAP.values()}
TICKER_TO_BANK = {v["ticker"]: v["bank"] for v in BANK_STOCK_MAP.values()}
APPID_TO_BANK = {k: v["bank"] for k, v in BANK_STOCK_MAP.items()}


def crawl_stock_data(start_date="2024-01-01", end_date=None):
    """
    Crawl giá cổ phiếu hàng ngày cho 11 ngân hàng từ Yahoo Finance.

    Returns:
        DataFrame: columns = [date, ticker, bank_name, open, high, low, close, volume, pct_change]
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 70)
    print("📈 CRAWL GIÁ CỔ PHIẾU NGÂN HÀNG")
    print("=" * 70)
    print(f"   📅 Từ: {start_date}  →  Đến: {end_date}")
    print(f"   🏦 Số ngân hàng: {len(BANK_NAME_TO_TICKER)}\n")

    all_data = []

    for bank_name, ticker in BANK_NAME_TO_TICKER.items():
        try:
            print(f"   ⏳ {bank_name} ({ticker})...", end=" ")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if len(df) == 0:
                print("❌ Không có dữ liệu")
                continue

            df = df.reset_index()
            df["ticker"] = ticker.replace(".VN", "")
            df["bank_name"] = bank_name

            # Rename columns
            df = df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Remove timezone info
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

            # Calculate daily % change
            df["pct_change"] = df["close"].pct_change() * 100

            # Keep relevant columns
            df = df[
                [
                    "date",
                    "ticker",
                    "bank_name",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "pct_change",
                ]
            ]

            all_data.append(df)
            print(f"✅ {len(df)} ngày giao dịch")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Lỗi: {e}")

    if not all_data:
        print("\n❌ Không crawl được dữ liệu nào!")
        return pd.DataFrame()

    stock_df = pd.concat(all_data, ignore_index=True)

    print(f"\n✅ Tổng cộng: {len(stock_df):,} records")
    print(
        f"   📅 Từ {stock_df['date'].min().date()} đến {stock_df['date'].max().date()}"
    )

    return stock_df


def merge_sentiment_stock_weekly(reviews_path=None, stock_df=None):
    """
    Merge sentiment data với stock data theo TUẦN.

    Cho mỗi tuần, tính:
    - Sentiment: avg_rating, review_count, positive_ratio, negative_ratio
    - Stock: avg_close, weekly_return, avg_volume

    Returns:
        DataFrame: weekly merged data per bank
    """
    print("\n" + "=" * 70)
    print("🔗 MERGE SENTIMENT × STOCK DATA (WEEKLY)")
    print("=" * 70)

    # Load reviews
    if reviews_path is None:
        reviews_path = f"{OUTPUT_DIR}/reviews_with_sentiment.csv"

    reviews = pd.read_csv(reviews_path)
    reviews["at"] = pd.to_datetime(reviews["at"])

    # Add bank_name if missing
    if "bank_name" not in reviews.columns:
        reviews["bank_name"] = reviews["appId"].map(APPID_TO_BANK)

    # Filter: only banks with stock data (exclude Agribank)
    reviews = reviews[reviews["bank_name"].isin(BANK_NAME_TO_TICKER.keys())]

    print(f"   📝 Reviews (excl. Agribank): {len(reviews):,}")

    # --- Weekly sentiment aggregation ---
    reviews["week"] = reviews["at"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly_sentiment = (
        reviews.groupby(["week", "bank_name"])
        .agg(
            review_count=("content", "size"),
            avg_rating=("score", "mean"),
            positive_count=("sentiment", lambda x: (x == "positive").sum()),
            negative_count=("sentiment", lambda x: (x == "negative").sum()),
        )
        .reset_index()
    )

    weekly_sentiment["positive_ratio"] = (
        weekly_sentiment["positive_count"] / weekly_sentiment["review_count"] * 100
    ).round(2)
    weekly_sentiment["negative_ratio"] = (
        weekly_sentiment["negative_count"] / weekly_sentiment["review_count"] * 100
    ).round(2)
    weekly_sentiment["avg_rating"] = weekly_sentiment["avg_rating"].round(3)

    print(f"   📊 Weekly sentiment records: {len(weekly_sentiment):,}")

    # --- Weekly stock aggregation ---
    if stock_df is None or len(stock_df) == 0:
        print("   ❌ Không có stock data!")
        return weekly_sentiment

    stock_df["week"] = stock_df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly_stock = (
        stock_df.groupby(["week", "bank_name"])
        .agg(
            avg_close=("close", "mean"),
            week_open=("open", "first"),
            week_close=("close", "last"),
            avg_volume=("volume", "mean"),
            trading_days=("date", "size"),
        )
        .reset_index()
    )

    # Weekly return (%)
    weekly_stock["weekly_return"] = (
        (weekly_stock["week_close"] - weekly_stock["week_open"])
        / weekly_stock["week_open"]
        * 100
    ).round(3)

    weekly_stock["avg_close"] = weekly_stock["avg_close"].round(0)
    weekly_stock["avg_volume"] = weekly_stock["avg_volume"].round(0)

    print(f"   📈 Weekly stock records: {len(weekly_stock):,}")

    # --- MERGE ---
    merged = pd.merge(
        weekly_sentiment,
        weekly_stock,
        on=["week", "bank_name"],
        how="inner",
    )

    print(f"   🔗 Merged records: {len(merged):,}")
    print(f"   🏦 Banks: {merged['bank_name'].nunique()}")
    print(
        f"   📅 Period: {merged['week'].min().date()} → {merged['week'].max().date()}"
    )

    return merged


def calculate_correlations(merged_df):
    """
    Tính Pearson & Spearman correlation cho mỗi ngân hàng.

    Pairs:
    - avg_rating vs weekly_return
    - avg_rating vs avg_close
    - positive_ratio vs weekly_return
    - negative_ratio vs weekly_return
    - review_count vs avg_volume

    Returns:
        DataFrame: correlation results per bank
    """
    from scipy import stats

    print("\n" + "=" * 70)
    print("📐 TÍNH TƯƠNG QUAN (CORRELATION ANALYSIS)")
    print("=" * 70)

    pairs = [
        ("avg_rating", "weekly_return", "Rating TB ↔ Return tuần"),
        ("avg_rating", "avg_close", "Rating TB ↔ Giá CP trung bình"),
        ("positive_ratio", "weekly_return", "% Tích cực ↔ Return tuần"),
        ("negative_ratio", "weekly_return", "% Tiêu cực ↔ Return tuần"),
        ("review_count", "avg_volume", "Số reviews ↔ Khối lượng GD"),
    ]

    results = []

    for bank in sorted(merged_df["bank_name"].unique()):
        bank_data = merged_df[merged_df["bank_name"] == bank].dropna()

        if len(bank_data) < 10:
            print(f"   ⚠️ {bank}: chỉ {len(bank_data)} tuần → bỏ qua")
            continue

        for col_x, col_y, label in pairs:
            x = bank_data[col_x].values
            y = bank_data[col_y].values

            # Remove any remaining NaN/inf
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) < 10:
                continue

            # Pearson
            pearson_r, pearson_p = stats.pearsonr(x, y)

            # Spearman
            spearman_r, spearman_p = stats.spearmanr(x, y)

            results.append(
                {
                    "bank_name": bank,
                    "variable_x": col_x,
                    "variable_y": col_y,
                    "label": label,
                    "n_weeks": len(x),
                    "pearson_r": round(pearson_r, 4),
                    "pearson_p": round(pearson_p, 4),
                    "spearman_r": round(spearman_r, 4),
                    "spearman_p": round(spearman_p, 4),
                    "significant_pearson": pearson_p < 0.05,
                    "significant_spearman": spearman_p < 0.05,
                }
            )

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        sig_count = results_df["significant_pearson"].sum()
        total = len(results_df)
        print(f"\n   📊 Tổng cặp phân tích: {total}")
        print(
            f"   ✅ Có ý nghĩa thống kê (p<0.05): {sig_count} ({sig_count/total*100:.1f}%)"
        )
        print(
            f"   ❌ Không có ý nghĩa: {total - sig_count} ({(total-sig_count)/total*100:.1f}%)"
        )

    return results_df


def granger_causality_test(merged_df, maxlag=4):
    """
    Granger Causality Test:
    Kiểm tra xem sentiment có "Granger-cause" stock return không?

    H0: Sentiment KHÔNG Granger-cause stock return
    H1: Sentiment CÓ Granger-cause stock return

    Returns:
        DataFrame: Granger test results per bank
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    print("\n" + "=" * 70)
    print("🔬 GRANGER CAUSALITY TEST")
    print("=" * 70)
    print(f"   Max lag: {maxlag} tuần\n")

    results = []

    pairs = [
        ("avg_rating", "weekly_return", "Rating → Return?"),
        ("negative_ratio", "weekly_return", "% Tiêu cực → Return?"),
    ]

    for bank in sorted(merged_df["bank_name"].unique()):
        bank_data = (
            merged_df[merged_df["bank_name"] == bank]
            .sort_values("week")
            .dropna(subset=["avg_rating", "weekly_return", "negative_ratio"])
        )

        if len(bank_data) < maxlag + 15:
            print(f"   ⚠️ {bank}: thiếu dữ liệu ({len(bank_data)} tuần) → bỏ qua")
            continue

        for col_x, col_y, label in pairs:
            try:
                test_data = bank_data[[col_y, col_x]].values

                # Suppress output
                import io
                import sys

                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                gc_result = grangercausalitytests(test_data, maxlag=maxlag)
                sys.stdout = old_stdout

                # Get best (lowest) p-value across lags
                best_p = 1.0
                best_lag = 1
                for lag in range(1, maxlag + 1):
                    p_val = gc_result[lag][0]["ssr_ftest"][1]
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag

                results.append(
                    {
                        "bank_name": bank,
                        "cause": col_x,
                        "effect": col_y,
                        "label": label,
                        "best_lag": best_lag,
                        "p_value": round(best_p, 4),
                        "significant": best_p < 0.05,
                        "interpretation": (
                            f"CÓ nhân quả Granger (lag={best_lag} tuần)"
                            if best_p < 0.05
                            else "Không có nhân quả Granger"
                        ),
                    }
                )

                status = "✅" if best_p < 0.05 else "❌"
                print(f"   {status} {bank}: {label} p={best_p:.4f} (lag={best_lag})")

            except Exception as e:
                print(f"   ⚠️ {bank} - {label}: {str(e)[:50]}")

    return pd.DataFrame(results)


def main():
    """Main pipeline: Crawl → Merge → Correlate → Save"""
    print("=" * 70)
    print("💹 FINANCIAL CORRELATION ANALYSIS")
    print("   Phân tích tương quan Sentiment × Giá cổ phiếu")
    print("=" * 70 + "\n")

    # 1. Crawl stock data
    stock_df = crawl_stock_data(start_date="2024-01-01")

    if len(stock_df) == 0:
        print("❌ Không có dữ liệu stock. Dừng lại.")
        return

    # Save raw stock data
    stock_path = f"{OUTPUT_DIR}/stock_prices.csv"
    stock_df.to_csv(stock_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved: {stock_path}")

    # 2. Merge with sentiment
    merged = merge_sentiment_stock_weekly(stock_df=stock_df)

    merged_path = f"{OUTPUT_DIR}/sentiment_stock_weekly.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {merged_path}")

    # 3. Correlation analysis
    corr_df = calculate_correlations(merged)

    corr_path = f"{OUTPUT_DIR}/correlation_results.csv"
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {corr_path}")

    # 4. Granger Causality
    try:
        granger_df = granger_causality_test(merged, maxlag=4)

        granger_path = f"{OUTPUT_DIR}/granger_causality.csv"
        granger_df.to_csv(granger_path, index=False, encoding="utf-8-sig")
        print(f"💾 Saved: {granger_path}")
    except ImportError:
        print("⚠️ statsmodels chưa cài. Bỏ qua Granger test.")
        print("   Cài bằng: pip install statsmodels")
        granger_df = pd.DataFrame()

    # 5. Summary
    print("\n" + "=" * 70)
    print("🎉 FINANCIAL ANALYSIS HOÀN THÀNH!")
    print("=" * 70)
    print(
        f"""
📁 FILES CREATED:
   • {stock_path} (giá CP hàng ngày)
   • {merged_path} (sentiment × stock weekly)
   • {corr_path} (Pearson & Spearman correlation)
   • {f'{OUTPUT_DIR}/granger_causality.csv' if len(granger_df) > 0 else 'Granger: SKIPPED'}

📊 SUMMARY:
   • Stock data: {len(stock_df):,} daily records
   • Merged weekly: {len(merged):,} records
   • Correlation pairs: {len(corr_df):,}
   • Significant (p<0.05): {corr_df['significant_pearson'].sum() if len(corr_df) > 0 else 0}
"""
    )


if __name__ == "__main__":
    main()
