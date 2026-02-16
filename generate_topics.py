# -*- coding: utf-8 -*-
"""
GENERATE TOPICS: Phân loại chủ đề cho reviews đã có
=====================================================
Script chạy topic categorization trên reviews_with_sentiment.csv
và lưu kết quả ra output/reviews_with_topics.csv + output/topic_statistics.csv
"""

import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topic_categorizer import BankingTopicCategorizer

# ========================================
# CONFIG
# ========================================
INPUT_FILE = "./output/reviews_with_sentiment.csv"
OUTPUT_REVIEWS = "./output/reviews_with_topics.csv"
OUTPUT_TOPIC_STATS = "./output/topic_statistics.csv"
OUTPUT_TOPIC_BANK = "./output/topic_bank_matrix.csv"
OUTPUT_TOPIC_SENTIMENT = "./output/topic_sentiment_bank.csv"

# Bank name mapping
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


def main():
    print("=" * 70)
    print("🏷️  TOPIC CATEGORIZATION - MOBILE BANKING REVIEWS")
    print("=" * 70)

    # 1. Load data
    print(f"\n📂 Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Loaded {len(df):,} reviews")

    # Add bank_name if missing
    if "bank_name" not in df.columns and "appId" in df.columns:
        df["bank_name"] = df["appId"].map(APP_NAMES)

    # 2. Run topic categorization
    print("\n" + "=" * 70)
    categorizer = BankingTopicCategorizer()
    df = categorizer.categorize_dataframe(df, text_column="content", use_cleaned=True)

    # 3. Save reviews with topics
    print(f"💾 Saving reviews with topics to: {OUTPUT_REVIEWS}")
    # Convert list column to string for CSV compatibility
    df_save = df.copy()
    df_save["topic_all"] = df_save["topic_all"].apply(
        lambda x: "|".join(x) if x else "Khác"
    )
    df_save.to_csv(OUTPUT_REVIEWS, index=False, encoding="utf-8-sig")
    print(f"   ✅ Saved {len(df_save):,} reviews")

    # 4. Generate topic statistics
    print(f"\n💾 Saving topic statistics to: {OUTPUT_TOPIC_STATS}")
    topic_stats = categorizer.get_topic_stats(df)
    topic_stats.to_csv(OUTPUT_TOPIC_STATS, index=False, encoding="utf-8-sig")
    print(f"   ✅ Saved {len(topic_stats)} topics")
    print(topic_stats.to_string(index=False))

    # 5. Generate topic × bank matrix
    print(f"\n💾 Saving topic-bank matrix to: {OUTPUT_TOPIC_BANK}")
    topic_bank = categorizer.get_topic_bank_matrix(df)
    topic_bank.to_csv(OUTPUT_TOPIC_BANK, encoding="utf-8-sig")
    print(f"   ✅ Saved {topic_bank.shape[0]} topics × {topic_bank.shape[1]} banks")

    # 6. Generate topic × sentiment × bank
    print(f"\n💾 Saving topic-sentiment-bank to: {OUTPUT_TOPIC_SENTIMENT}")
    topic_sentiment = categorizer.get_topic_sentiment_bank(df)
    topic_sentiment.to_csv(OUTPUT_TOPIC_SENTIMENT, encoding="utf-8-sig")
    print(f"   ✅ Saved")

    # 7. Summary
    print("\n" + "=" * 70)
    print("🎉 TOPIC CATEGORIZATION HOÀN THÀNH!")
    print("=" * 70)
    print(
        f"""
📁 FILES CREATED:
   • {OUTPUT_REVIEWS} ({len(df_save):,} reviews)
   • {OUTPUT_TOPIC_STATS} (topic statistics)
   • {OUTPUT_TOPIC_BANK} (topic × bank matrix)
   • {OUTPUT_TOPIC_SENTIMENT} (topic × sentiment × bank)

📊 SUMMARY:
   • Total reviews: {len(df):,}
   • Topics found: {len(topic_stats)}
   • Multi-topic reviews: {(df['topic_count'] > 1).sum():,}
   • Uncategorized: {(df['topic_primary'] == 'Khác').sum():,}
"""
    )


if __name__ == "__main__":
    main()
