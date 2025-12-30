# src/eda_main.py
import json
from config import DATA_FILE, FIG_DIR, SUMMARY_DIR, OUTPUT_DIR
from helpers import (
    load_data, overview, basic_stats, missing_report,
    fill_numeric, detect_outliers_zscore,
    corr_heatmap, plot_distribution,
    boxplot_col, scatter_pair
)

def ensure_dirs():
    """Create output folders if not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # 1) Ensure output folders exist
    ensure_dirs()
    print("Output folders ready.")

    # 2) Load dataset
    print("\nLoading dataset...")
    df = load_data(DATA_FILE)
    print(f"Dataset loaded. Shape: {df.shape}")

    # 3) Dataset overview
    print("\nGenerating dataset overview...")
    ov = overview(df)
    with open(SUMMARY_DIR / "overview.json", "w", encoding="utf-8") as f:
        json.dump(ov, f, indent=2)
    print("Overview saved.")

    # 4) Basic statistics
    print("\nCalculating basic statistics...")
    stats = basic_stats(df)
    stats.to_csv(SUMMARY_DIR / "basic_stats.csv")
    print("Basic statistics saved.")

    # 5) Missing values report
    print("\nChecking missing values...")
    miss = missing_report(df)
    miss.to_csv(SUMMARY_DIR / "missing_report.csv")
    print("Missing report saved.")

    # 6) Fill missing numeric values
    print("\nFilling missing numeric values...")
    df_clean = fill_numeric(df.copy(), strategy="median")
    df_clean.to_csv(OUTPUT_DIR / "data_cleaned.csv", index=False)
    print("Cleaned dataset saved.")

    # 7) Outlier detection
    print("\nDetecting outliers using z-score...")
    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    outliers = detect_outliers_zscore(df_clean, numeric_cols)
    with open(SUMMARY_DIR / "outliers_indices.txt", "w", encoding="utf-8") as f:
        for idx in outliers:
            f.write(f"{idx}\n")
    print(f"Outliers found: {len(outliers)} (saved to outliers_indices.txt)")

    # 8) Correlation heatmap
    print("\nGenerating correlation heatmap...")
    corr = corr_heatmap(df_clean, FIG_DIR)
    corr.to_csv(SUMMARY_DIR / "correlation_matrix.csv")
    print("Correlation heatmap saved.")

    # 9) Plot distribution and boxplot for each numeric column
    print("\nGenerating distribution & boxplots...")
    for col in numeric_cols:
        print(f"  → {col}")
        plot_distribution(df_clean, col, FIG_DIR)
        boxplot_col(df_clean, col, FIG_DIR)

    # 10) Scatter plots for top correlated pairs
    print("\nGenerating scatter plots for correlated pairs...")
    corr_flat = corr.abs().unstack().sort_values(ascending=False)

    # Remove self correlations (A,A)
    corr_flat = corr_flat[corr_flat.index.get_level_values(0) != corr_flat.index.get_level_values(1)]

    # Pick top unique pairs
    used = set()
    pairs = []
    for (a, b), value in corr_flat.items():
        if (b, a) in used:
            continue
        pairs.append((a, b))
        used.add((a, b))
        used.add((b, a))
        if len(pairs) >= 5:
            break

    for x, y in pairs:
        print(f"  → Scatter: {x} vs {y}")
        scatter_pair(df_clean, x, y, FIG_DIR)

    print("\nEDA COMPLETE! Check the 'outputs' folder.")


if __name__ == "__main__":
    main()
