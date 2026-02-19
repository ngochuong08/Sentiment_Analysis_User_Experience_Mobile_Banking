# -*- coding: utf-8 -*-
"""
STOCK RETURN PREDICTION: ABLATION STUDY
=========================================
So sánh ML model dự báo weekly stock return:
  - Model A (Baseline): Chỉ dùng biến tài chính (lag returns, volume...)
  - Model B (+ Sentiment): Thêm avg_rating, negative_ratio, review_count

Nếu Model B tốt hơn → Sentiment có giá trị thông tin bổ sung.
Nếu Model B không tốt hơn → Thông tin sentiment đã được phản ánh trong giá.

Author: KLTN - Sentiment Analysis Mobile Banking
"""

import pandas as pd
import numpy as np
import os
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

OUTPUT_DIR = "./output"


# ========================================
# 1. FEATURE ENGINEERING
# ========================================
def build_features(df):
    """
    Tạo features cho ML prediction từ weekly merged data.

    Financial features (Model A):
      - lag_return_1w, lag_return_2w, lag_return_3w, lag_return_4w
      - lag_volume_change_1w
      - rolling_return_4w (moving average 4 tuần)
      - return_volatility_4w (độ biến động 4 tuần)

    Sentiment features (Model B = A + these):
      - avg_rating
      - negative_ratio
      - positive_ratio
      - review_count
      - lag_rating_1w, lag_neg_ratio_1w
      - rating_change_1w
      - sentiment_momentum_4w (rolling avg rating 4 tuần)
    """
    print("=" * 70)
    print("🔧 FEATURE ENGINEERING")
    print("=" * 70)

    all_features = []

    for bank in sorted(df["bank_name"].unique()):
        bank_data = df[df["bank_name"] == bank].sort_values("week").copy()

        if len(bank_data) < 15:
            print(f"   ⚠️ {bank}: chỉ {len(bank_data)} tuần → bỏ qua")
            continue

        # --- TARGET ---
        bank_data["target"] = bank_data["weekly_return"]

        # --- FINANCIAL FEATURES (Model A) ---
        # Lag returns
        for lag in [1, 2, 3, 4]:
            bank_data[f"lag_return_{lag}w"] = bank_data["weekly_return"].shift(lag)

        # Volume change
        bank_data["lag_volume_change_1w"] = bank_data["avg_volume"].pct_change() * 100

        # Rolling stats (4 weeks)
        bank_data["rolling_return_4w"] = bank_data["weekly_return"].rolling(4).mean()
        bank_data["return_volatility_4w"] = bank_data["weekly_return"].rolling(4).std()

        # --- SENTIMENT FEATURES (Model B adds these) ---
        # Current sentiment
        bank_data["sent_avg_rating"] = bank_data["avg_rating"]
        bank_data["sent_negative_ratio"] = bank_data["negative_ratio"]
        bank_data["sent_positive_ratio"] = bank_data["positive_ratio"]
        bank_data["sent_review_count"] = bank_data["review_count"]

        # Lag sentiment (previous week's sentiment → predict this week's return)
        bank_data["sent_lag_rating_1w"] = bank_data["avg_rating"].shift(1)
        bank_data["sent_lag_neg_ratio_1w"] = bank_data["negative_ratio"].shift(1)

        # Rating change
        bank_data["sent_rating_change_1w"] = bank_data["avg_rating"].diff()

        # Sentiment momentum (rolling 4 weeks)
        bank_data["sent_momentum_4w"] = bank_data["avg_rating"].rolling(4).mean()

        bank_data["bank_name_encoded"] = bank  # keep for later

        all_features.append(bank_data)

    result = pd.concat(all_features, ignore_index=True)

    # Drop rows with NaN from lag/rolling
    before = len(result)
    result = result.dropna()
    after = len(result)
    print(f"   📊 Total rows: {before:,} → after dropna: {after:,}")
    print(f"   🏦 Banks: {result['bank_name'].nunique()}")

    return result


# ========================================
# 2. TRAIN & EVALUATE MODELS
# ========================================
def train_and_compare(feature_df):
    """
    Train Model A (financial only) vs Model B (financial + sentiment).
    Use TimeSeriesSplit for proper evaluation.

    Models: Linear Regression, Random Forest, XGBoost
    Metrics: RMSE, MAE, R²
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 70)
    print("🤖 TRAINING ML MODELS")
    print("=" * 70)

    # Define feature sets
    financial_features = [
        "lag_return_1w",
        "lag_return_2w",
        "lag_return_3w",
        "lag_return_4w",
        "lag_volume_change_1w",
        "rolling_return_4w",
        "return_volatility_4w",
    ]

    sentiment_features = [
        "sent_avg_rating",
        "sent_negative_ratio",
        "sent_positive_ratio",
        "sent_review_count",
        "sent_lag_rating_1w",
        "sent_lag_neg_ratio_1w",
        "sent_rating_change_1w",
        "sent_momentum_4w",
    ]

    features_A = financial_features
    features_B = financial_features + sentiment_features

    target = "target"

    print(f"   📐 Model A features: {len(features_A)} (financial only)")
    print(f"   📐 Model B features: {len(features_B)} (financial + sentiment)")

    # Models to compare
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    # Prepare data (sort by time for proper splitting)
    feature_df = feature_df.sort_values("week").reset_index(drop=True)

    # Replace infinite values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna(subset=features_B + [target])

    X_A = feature_df[features_A].values
    X_B = feature_df[features_B].values
    y = feature_df[target].values

    print(f"   📊 Dataset size: {len(y)} samples")

    # TimeSeriesSplit (5-fold)
    tscv = TimeSeriesSplit(n_splits=5)

    results = []

    for model_name, model_template in models.items():
        print(f"\n   🔄 {model_name}...")

        for variant, X, n_features in [
            ("A (Financial)", X_A, len(features_A)),
            ("B (+ Sentiment)", X_B, len(features_B)),
        ]:
            all_y_true = []
            all_y_pred = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Clone model for each fold
                from sklearn.base import clone

                model = clone(model_template)

                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

                all_y_true.extend(y_test)
                all_y_pred.extend(preds)

            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)

            rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
            mae = mean_absolute_error(all_y_true, all_y_pred)
            r2 = r2_score(all_y_true, all_y_pred)

            label = f"{model_name} - {variant}"
            print(f"      {variant}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            results.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "n_features": n_features,
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "r2": round(r2, 4),
                }
            )

    return pd.DataFrame(results), feature_df, features_A, features_B


# ========================================
# 3. FEATURE IMPORTANCE (SHAP-like)
# ========================================
def compute_feature_importance(feature_df, features_B):
    """
    Train best model (Gradient Boosting) trên toàn bộ data,
    trả về feature importance.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 70)
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    target = "target"

    X = feature_df[features_B].values
    y = feature_df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_scaled, y)

    importance = pd.DataFrame(
        {
            "feature": features_B,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance["importance_pct"] = (
        importance["importance"] / importance["importance"].sum() * 100
    ).round(2)

    # Classify features
    importance["type"] = importance["feature"].apply(
        lambda x: "Sentiment" if x.startswith("sent_") else "Financial"
    )

    print("\n   Top features:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance_pct"])
        print(
            f"   {row['importance_pct']:6.2f}% {bar} {row['feature']} [{row['type']}]"
        )

    # Summary
    sent_total = importance[importance["type"] == "Sentiment"]["importance_pct"].sum()
    fin_total = importance[importance["type"] == "Financial"]["importance_pct"].sum()
    print(f"\n   📊 Financial contribution: {fin_total:.1f}%")
    print(f"   📊 Sentiment contribution: {sent_total:.1f}%")

    return importance


# ========================================
# 4. PER-BANK ANALYSIS
# ========================================
def per_bank_comparison(feature_df, features_A, features_B):
    """
    So sánh Model A vs B cho TỪNG ngân hàng.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone

    print("\n" + "=" * 70)
    print("🏦 PER-BANK COMPARISON (Gradient Boosting)")
    print("=" * 70)

    target = "target"
    results = []

    for bank in sorted(feature_df["bank_name"].unique()):
        bank_data = feature_df[feature_df["bank_name"] == bank].sort_values("week")

        if len(bank_data) < 30:
            print(f"   ⚠️ {bank}: {len(bank_data)} samples → skip")
            continue

        X_A = bank_data[features_A].values
        X_B = bank_data[features_B].values
        y = bank_data[target].values

        n_splits = min(3, len(bank_data) // 15)
        if n_splits < 2:
            continue

        tscv = TimeSeriesSplit(n_splits=n_splits)

        model_template = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=3,
            random_state=42,
        )

        for variant, X in [("A", X_A), ("B", X_B)]:
            all_true, all_pred = [], []

            for train_idx, test_idx in tscv.split(X):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_te = scaler.transform(X[test_idx])

                model = clone(model_template)
                model.fit(X_tr, y[train_idx])
                preds = model.predict(X_te)

                all_true.extend(y[test_idx])
                all_pred.extend(preds)

            rmse = np.sqrt(mean_squared_error(all_true, all_pred))
            r2 = r2_score(all_true, all_pred)

            results.append(
                {
                    "bank_name": bank,
                    "variant": variant,
                    "n_samples": len(bank_data),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                }
            )

        # Print comparison
        a = [r for r in results if r["bank_name"] == bank and r["variant"] == "A"][-1]
        b = [r for r in results if r["bank_name"] == bank and r["variant"] == "B"][-1]
        delta_rmse = b["rmse"] - a["rmse"]
        delta_r2 = b["r2"] - a["r2"]
        better = (
            "✅ B tốt hơn"
            if delta_r2 > 0.01
            else ("❌ A tốt hơn" if delta_r2 < -0.01 else "⚖️ Tương đương")
        )
        print(
            f"   {bank:15s} | A: R²={a['r2']:.4f} RMSE={a['rmse']:.4f} | B: R²={b['r2']:.4f} RMSE={b['rmse']:.4f} | ΔR²={delta_r2:+.4f} | {better}"
        )

    return pd.DataFrame(results)


# ========================================
# 5. MAIN PIPELINE
# ========================================
def main():
    print("=" * 70)
    print("🎯 STOCK RETURN PREDICTION: ABLATION STUDY")
    print("   Model A (Financial) vs Model B (Financial + Sentiment)")
    print("=" * 70 + "\n")

    # Load merged weekly data
    merged_path = f"{OUTPUT_DIR}/sentiment_stock_weekly.csv"
    if not os.path.exists(merged_path):
        print("❌ Chưa có sentiment_stock_weekly.csv!")
        print("   Chạy: python financial_analysis.py")
        return

    df = pd.read_csv(merged_path)
    df["week"] = pd.to_datetime(df["week"])
    print(f"📊 Loaded: {len(df):,} weekly records, {df['bank_name'].nunique()} banks\n")

    # 1. Feature engineering
    feature_df = build_features(df)

    # 2. Train & compare (all banks pooled)
    comparison_df, feature_df, feat_A, feat_B = train_and_compare(feature_df)

    # Save comparison
    comp_path = f"{OUTPUT_DIR}/stock_prediction_comparison.csv"
    comparison_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved: {comp_path}")

    # 3. Feature importance
    importance_df = compute_feature_importance(feature_df, feat_B)

    imp_path = f"{OUTPUT_DIR}/feature_importance.csv"
    importance_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {imp_path}")

    # 4. Per-bank comparison
    bank_comp_df = per_bank_comparison(feature_df, feat_A, feat_B)

    bank_path = f"{OUTPUT_DIR}/stock_prediction_per_bank.csv"
    bank_comp_df.to_csv(bank_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {bank_path}")

    # 5. Summary
    print("\n" + "=" * 70)
    print("🎉 ABLATION STUDY HOÀN THÀNH!")
    print("=" * 70)

    # Overall comparison
    print("\n📊 TỔNG KẾT (All banks pooled):")
    print("-" * 65)
    for _, row in comparison_df.iterrows():
        print(
            f"   {row['model']:25s} {row['variant']:20s} | RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} R²={row['r2']:.4f}"
        )

    # Best models
    best_A = (
        comparison_df[comparison_df["variant"].str.contains("Financial")]
        .sort_values("rmse")
        .iloc[0]
    )
    best_B = (
        comparison_df[comparison_df["variant"].str.contains("Sentiment")]
        .sort_values("rmse")
        .iloc[0]
    )

    delta_rmse = best_B["rmse"] - best_A["rmse"]
    delta_r2 = best_B["r2"] - best_A["r2"]

    print(
        f"\n🏆 Best Model A: {best_A['model']} | RMSE={best_A['rmse']:.4f}, R²={best_A['r2']:.4f}"
    )
    print(
        f"🏆 Best Model B: {best_B['model']} | RMSE={best_B['rmse']:.4f}, R²={best_B['r2']:.4f}"
    )
    print(
        f"   ΔRMSE = {delta_rmse:+.4f} ({'B tốt hơn' if delta_rmse < 0 else 'A tốt hơn'})"
    )
    print(
        f"   ΔR²   = {delta_r2:+.4f} ({'B tốt hơn' if delta_r2 > 0 else 'A tốt hơn'})"
    )

    # Sentiment contribution
    sent_pct = importance_df[importance_df["type"] == "Sentiment"][
        "importance_pct"
    ].sum()
    print(f"\n📊 Sentiment contribution to prediction: {sent_pct:.1f}%")

    # Per-bank summary
    bank_A = bank_comp_df[bank_comp_df["variant"] == "A"].set_index("bank_name")
    bank_B = bank_comp_df[bank_comp_df["variant"] == "B"].set_index("bank_name")
    banks_improved = 0
    common_banks = set(bank_A.index) & set(bank_B.index)
    for bank in common_banks:
        if bank_B.loc[bank, "r2"] > bank_A.loc[bank, "r2"] + 0.01:
            banks_improved += 1

    print(
        f"   Banks where sentiment improved R² > 1%: {banks_improved}/{len(common_banks)}"
    )

    # Conclusion
    if delta_r2 > 0.01:
        conclusion = "Sentiment CÓ giá trị bổ sung cho dự báo stock return"
    elif delta_r2 < -0.01:
        conclusion = (
            "Sentiment KHÔNG cải thiện dự báo — thông tin đã được phản ánh trong giá"
        )
    else:
        conclusion = "Sentiment có tác động MARGINAL — cải thiện không đáng kể"

    print(f"\n💡 KẾT LUẬN: {conclusion}")

    # Save summary as JSON
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(feature_df),
        "n_banks": int(feature_df["bank_name"].nunique()),
        "best_model_A": {
            "model": best_A["model"],
            "rmse": float(best_A["rmse"]),
            "r2": float(best_A["r2"]),
        },
        "best_model_B": {
            "model": best_B["model"],
            "rmse": float(best_B["rmse"]),
            "r2": float(best_B["r2"]),
        },
        "delta_rmse": float(delta_rmse),
        "delta_r2": float(delta_r2),
        "sentiment_contribution_pct": float(sent_pct),
        "banks_improved": banks_improved,
        "total_banks": len(common_banks),
        "conclusion": conclusion,
    }

    summary_path = f"{OUTPUT_DIR}/stock_prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {summary_path}")

    print(
        f"""
📁 FILES CREATED:
   • {comp_path}
   • {imp_path}
   • {bank_path}
   • {summary_path}
"""
    )


if __name__ == "__main__":
    main()
