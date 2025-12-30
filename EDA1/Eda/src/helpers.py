import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set default Seaborn style
sns.set(style="whitegrid", rc={"figure.figsize": (8, 5)})

def load_data(path):
    """Load CSV into a pandas DataFrame."""
    return pd.read_csv(path)

def overview(df):
    """Return shape, datatypes, and missing values."""
    # Convert to JSON-serializable types:
    # - shape as list
    # - dtypes as strings
    # - missing counts as Python ints
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
    return {
        "shape": list(df.shape),
        "dtypes": dtypes,
        "missing": missing
    }

def basic_stats(df):
    """Return descriptive statistics."""
    return df.describe(include='all').T

def missing_report(df):
    """Missing values count and percentage."""
    percent_missing = df.isnull().mean() * 100
    return pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": percent_missing
    })

def fill_numeric(df, strategy="median"):
    """Fill missing numeric values using Imputer."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        raise ImportError("scikit-learn is required for fill_numeric; please install it via 'pip install scikit-learn'")
    imputer = SimpleImputer(strategy=strategy)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def detect_outliers_zscore(df, cols, z_thresh=3.0):
    """Return row indices where any numeric column has z-score > threshold."""
    # Lazy import to avoid import-time failure if scipy is not installed
    try:
        from scipy import stats
    except Exception:
        raise ImportError("scipy is required for detect_outliers_zscore; please install it via 'pip install scipy'")
    outliers = set()
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            z = np.abs(stats.zscore(df[col].astype(float), nan_policy='omit'))
            indices = np.where(z > z_thresh)[0]
            outliers.update(indices.tolist())
    return sorted(list(outliers))

def save_fig(fig, filename, figdir):
    """Save figure to folder."""
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

def corr_heatmap(df, figdir, filename="corr_heatmap.png"):
    """Generate correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    save_fig(fig, filename, figdir)
    return corr

def plot_distribution(df, col, figdir):
    """Plot distribution (histogram + KDE)."""
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    save_fig(fig, f"dist_{col}.png", figdir)

def boxplot_col(df, col, figdir):
    """Plot boxplot for a column."""
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    save_fig(fig, f"box_{col}.png", figdir)

def scatter_pair(df, x, y, figdir):
    """Scatter plot between two columns."""
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x], y=df[y], ax=ax)
    ax.set_title(f"{x} vs {y}")
    save_fig(fig, f"scatter_{x}_{y}.png", figdir)
