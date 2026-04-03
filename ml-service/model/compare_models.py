"""
model/compare_models.py
=======================
Systematic comparison of XGBoost vs LightGBM on the LoanSense synthetic dataset.

Both models are trained under **identical conditions**:
  - same dataset (synthetic_loan_data.csv)
  - same preprocessing (CategoricalEncoder → StandardScaler)
  - same feature set (11 features)
  - same train/test split (random_state=42, stratify=y)
  - equivalent hyper-parameters (n_estimators=300, max_depth=6, lr=0.05, …)

Run from the ml-service root:
    python3 model/compare_models.py              # train fresh, print table
    python3 model/compare_models.py --no-plots   # skip matplotlib output

Output
------
  • Console: per-model metrics + comparison table
  • Files:   model/model_xgb.pkl, model/model_lgbm.pkl  (saved after training)
  • Plots:   model/roc_curves.png, model/feature_importance.png  (if matplotlib available)
"""

import sys
import os
import time
import argparse
import warnings
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    roc_curve,
)
from scipy.stats import pearsonr

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.preprocess import (
    CategoricalEncoder,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "synthetic_loan_data.csv")

# Suppress LightGBM/sklearn feature-name mismatch warnings that appear when
# predicting with numpy arrays on a model fitted via the sklearn API.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed",
    category=UserWarning,
)

# ── optional imports ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("⚠️  xgboost not installed — install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    print("⚠️  lightgbm not installed — install with: pip install lightgbm")

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for servers
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, StandardScaler, CategoricalEncoder]:
    """
    Load synthetic_loan_data.csv, preprocess, and return the same train/test
    arrays that the individual training scripts use.

    Returns
    -------
    X_train_s, X_test_s  : scaled feature arrays
    y_train, y_test      : label arrays
    X_test_raw           : unscaled test features (for SHAP)
    feature_names        : list of feature column names
    scaler               : fitted StandardScaler
    encoder              : fitted CategoricalEncoder
    """
    print("📂 Loading data …")
    df = pd.read_csv(DATA_PATH)
    print(f"   Rows: {len(df):,} | Approval rate: {df['eligible'].mean():.1%}")

    df = df.rename(columns={
        "loan_amount":          "requested_loan_amount",
        "existing_emis":        "total_existing_emi",
        "emi_to_income_ratio":  "dti_ratio",
        "loan_to_income_ratio": "loan_to_income",
    })
    if "employment_tenure_months" not in df.columns:
        df["employment_tenure_months"] = df["work_experience_years"] * 12
    if "has_coapplicant" not in df.columns:
        df["has_coapplicant"] = 0

    encoder = CategoricalEncoder()
    df_enc  = encoder.fit_transform_df(df)

    X = df_enc[FEATURE_COLUMNS].values
    y = df_enc["eligible"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}\n")
    return X_train_s, X_test_s, y_train, y_test, X_test, FEATURE_COLUMNS, scaler, encoder


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, float]:
    """Train XGBoost classifier and return (model, training_time_seconds)."""
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is required. Install with: pip install xgboost")
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
    elapsed = time.perf_counter() - t0
    return model, elapsed


def train_lgbm(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, float]:
    """Train LightGBM classifier and return (model, training_time_seconds)."""
    if not _LGBM_AVAILABLE:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")
    model = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary", metric="binary_logloss",
        boosting_type="gbdt", num_leaves=31,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    elapsed = time.perf_counter() - t0
    return model, elapsed


def load_or_train_model(
    name: str,
    pkl_path: str,
    train_fn,
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, float]:
    """
    Load a pre-trained model from *pkl_path* if it exists, otherwise call
    *train_fn* to train it fresh and save the artifact.

    Returns (model, training_time_seconds).
    training_time is 0.0 when loading from disk.
    """
    if os.path.exists(pkl_path):
        print(f"📦 Loading {name} from {os.path.basename(pkl_path)} …")
        model = joblib.load(pkl_path)
        train_time = 0.0
    else:
        print(f"🚀 Training {name} …")
        model, train_time = train_fn(X_train_s, y_train, X_test_s, y_test)
        joblib.dump(model, pkl_path)
        print(f"   Saved → {os.path.basename(pkl_path)}  ({train_time:.2f}s)")
    return model, train_time


# ─────────────────────────────────────────────────────────────────────────────
# 3. Performance evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute accuracy, ROC-AUC, and F1 on the test set.

    Returns a dict with keys: accuracy, roc_auc, f1, proba (np.ndarray), pred (np.ndarray).
    """
    proba = model.predict_proba(X_test_s)[:, 1]
    pred  = model.predict(X_test_s)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc":  roc_auc_score(y_test, proba),
        "f1":       f1_score(y_test, pred),
        "proba":    proba,
        "pred":     pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference latency
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_latency(
    model: Any,
    X_test_s: np.ndarray,
    n_runs: int = 10,
) -> float:
    """
    Warm up the model then measure average prediction time over *n_runs*.

    Returns average wall-clock milliseconds to predict the full test set.
    """
    # warm-up (JIT / caching effects)
    model.predict_proba(X_test_s)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_proba(X_test_s)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)) * 1000  # → ms


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP feature importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(
    model: Any,
    X_test_s: np.ndarray,
    feature_names: list,
    sample_size: int = 500,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute mean absolute SHAP values and return a ranked DataFrame.

    Uses a random subsample of the test set to keep runtime short.
    Returns (shap_values_array, ranked_df).
    """
    if len(X_test_s) == 0:
        raise ValueError("X_test_s is empty; cannot compute SHAP values.")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test_s), size=min(sample_size, len(X_test_s)), replace=False)
    X_sample = X_test_s[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM TreeExplainer may return a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked   = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    return mean_abs, ranked


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prediction stability
# ─────────────────────────────────────────────────────────────────────────────

def compare_predictions(
    proba_xgb: np.ndarray,
    proba_lgbm: np.ndarray,
    pred_xgb: np.ndarray,
    pred_lgbm: np.ndarray,
) -> Dict[str, float]:
    """
    Quantify how similar the two models' outputs are.

    Returns dict with pearson_r, percent_agreement.
    """
    r, _ = pearsonr(proba_xgb, proba_lgbm)
    agreement = (pred_xgb == pred_lgbm).mean() * 100
    return {
        "pearson_r":         float(r),
        "percent_agreement": float(agreement),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _winner(val_xgb: float, val_lgbm: float, higher_is_better: bool = True) -> str:
    """Return '✅ XGB', '✅ LGBM', or '🤝 TIE' based on the metric direction."""
    diff = abs(val_xgb - val_lgbm)
    if diff < 1e-4:
        return "🤝 TIE"
    if higher_is_better:
        return "✅ XGB" if val_xgb > val_lgbm else "✅ LGBM"
    return "✅ XGB" if val_xgb < val_lgbm else "✅ LGBM"


def print_comparison_table(
    metrics_xgb: Dict[str, float],
    metrics_lgbm: Dict[str, float],
    latency_xgb: float,
    latency_lgbm: float,
    train_time_xgb: float,
    train_time_lgbm: float,
    stability: Dict[str, float],
) -> pd.DataFrame:
    """Print a formatted comparison table and return it as a DataFrame."""
    rows = [
        ("Accuracy",             metrics_xgb["accuracy"],  metrics_lgbm["accuracy"],  True),
        ("ROC-AUC",              metrics_xgb["roc_auc"],   metrics_lgbm["roc_auc"],   True),
        ("F1-Score",             metrics_xgb["f1"],        metrics_lgbm["f1"],        True),
        ("Inference latency (ms)", latency_xgb,            latency_lgbm,              False),
        ("Training time (s)",    train_time_xgb,           train_time_lgbm,           False),
    ]

    col_w = 26
    num_w = 10
    title_xgb  = "XGBoost"
    title_lgbm = "LightGBM"
    title_win  = "Winner"

    header = (
        f"{'Metric':<{col_w}}"
        f"{title_xgb:>{num_w}}"
        f"{title_lgbm:>{num_w}}"
        f"  {title_win}"
    )
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  MODEL COMPARISON: XGBoost  vs  LightGBM")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    table_rows = []
    for label, xgb_val, lgbm_val, higher_is_better in rows:
        # Hide training time row when both models were loaded from disk
        if train_time_xgb == 0.0 and train_time_lgbm == 0.0 and label == "Training time (s)":
            winner_str = "—  (loaded from disk)"
            xgb_str   = "—"
            lgbm_str  = "—"
        else:
            winner_str = _winner(xgb_val, lgbm_val, higher_is_better)
            xgb_str    = f"{xgb_val:.4f}"
            lgbm_str   = f"{lgbm_val:.4f}"
        print(f"{label:<{col_w}}{xgb_str:>{num_w}}{lgbm_str:>{num_w}}  {winner_str}")
        table_rows.append({
            "Metric":   label,
            "XGBoost":  xgb_val if xgb_str != "—" else None,
            "LightGBM": lgbm_val if lgbm_str != "—" else None,
            "Winner":   winner_str,
        })

    print(sep)
    print(f"\n  Prediction correlation (Pearson r): {stability['pearson_r']:.4f}")
    print(f"  Identical class predictions:        {stability['percent_agreement']:.1f}%")
    print(f"{'═' * len(header)}\n")

    return pd.DataFrame(table_rows)


def print_shap_comparison(ranked_xgb: pd.DataFrame, ranked_lgbm: pd.DataFrame) -> None:
    """Side-by-side SHAP feature ranking for both models."""
    top_n = min(len(ranked_xgb), len(ranked_lgbm), 11)
    col_w = 32
    num_w = 10

    header = (
        f"{'XGBoost feature':<{col_w}}{' SHAP':>{num_w}}"
        f"   {'LightGBM feature':<{col_w}}{' SHAP':>{num_w}}"
    )
    sep = "─" * len(header)

    print("SHAP Feature Importance Comparison (top features, mean |SHAP|)")
    print(sep)
    print(header)
    print(sep)
    for i in range(top_n):
        xgb_row  = ranked_xgb.iloc[i]
        lgbm_row = ranked_lgbm.iloc[i]
        xgb_label  = FEATURE_LABELS.get(xgb_row["feature"], xgb_row["feature"])
        lgbm_label = FEATURE_LABELS.get(lgbm_row["feature"], lgbm_row["feature"])
        print(
            f"{xgb_label:<{col_w}}{xgb_row['mean_abs_shap']:>{num_w}.4f}"
            f"   {lgbm_label:<{col_w}}{lgbm_row['mean_abs_shap']:>{num_w}.4f}"
        )

    # Rank consistency: how many of top-5 features are shared?
    top5_xgb  = set(ranked_xgb.head(5)["feature"])
    top5_lgbm = set(ranked_lgbm.head(5)["feature"])
    shared    = top5_xgb & top5_lgbm
    print(sep)
    print(f"  Top-5 shared features: {len(shared)}/5  → {sorted(shared)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Optional plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_test: np.ndarray,
    proba_xgb: np.ndarray,
    proba_lgbm: np.ndarray,
    auc_xgb: float,
    auc_lgbm: float,
    save_path: str,
) -> None:
    """Plot and save ROC curves for both models."""
    fpr_xgb,  tpr_xgb,  _ = roc_curve(y_test, proba_xgb)
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, proba_lgbm)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_xgb,  tpr_xgb,  lw=2, label=f"XGBoost  (AUC = {auc_xgb:.4f})",  color="#e06c75")
    ax.plot(fpr_lgbm, tpr_lgbm, lw=2, label=f"LightGBM (AUC = {auc_lgbm:.4f})", color="#61afef")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — XGBoost vs LightGBM")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📈 ROC curve saved → {save_path}")


def plot_feature_importance(
    ranked_xgb: pd.DataFrame,
    ranked_lgbm: pd.DataFrame,
    save_path: str,
) -> None:
    """Horizontal bar chart comparing mean |SHAP| for both models."""
    top_n = 11
    feats = list(dict.fromkeys(
        ranked_xgb.head(top_n)["feature"].tolist() +
        ranked_lgbm.head(top_n)["feature"].tolist()
    ))[:top_n]

    xgb_vals  = [ranked_xgb.set_index("feature").loc[f, "mean_abs_shap"]
                  if f in ranked_xgb["feature"].values else 0.0 for f in feats]
    lgbm_vals = [ranked_lgbm.set_index("feature").loc[f, "mean_abs_shap"]
                  if f in ranked_lgbm["feature"].values else 0.0 for f in feats]

    labels = [FEATURE_LABELS.get(f, f) for f in feats]
    y_pos  = np.arange(len(labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y_pos - height / 2, xgb_vals,  height, label="XGBoost",  color="#e06c75", alpha=0.85)
    ax.barh(y_pos + height / 2, lgbm_vals, height, label="LightGBM", color="#61afef", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance — XGBoost vs LightGBM")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📊 Feature importance chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────

def main(no_plots: bool = False) -> pd.DataFrame:
    """
    Run the full comparison pipeline and return the summary DataFrame.
    """
    if not _XGB_AVAILABLE or not _LGBM_AVAILABLE:
        missing = []
        if not _XGB_AVAILABLE:  missing.append("xgboost")
        if not _LGBM_AVAILABLE: missing.append("lightgbm")
        raise ImportError(f"Missing packages: {', '.join(missing)}. "
                          f"Install with: pip install {' '.join(missing)}")

    # ── data ──────────────────────────────────────────────────────────────────
    X_train_s, X_test_s, y_train, y_test, _, feature_names, scaler, encoder = load_data()

    # ── models ────────────────────────────────────────────────────────────────
    xgb_pkl  = os.path.join(MODEL_DIR, "model_xgb.pkl")
    lgbm_pkl = os.path.join(MODEL_DIR, "model_lgbm.pkl")

    model_xgb,  train_time_xgb  = load_or_train_model(
        "XGBoost",  xgb_pkl,  train_xgboost, X_train_s, y_train, X_test_s, y_test)
    model_lgbm, train_time_lgbm = load_or_train_model(
        "LightGBM", lgbm_pkl, train_lgbm,    X_train_s, y_train, X_test_s, y_test)

    # ── performance metrics ───────────────────────────────────────────────────
    print("📊 Evaluating XGBoost …")
    metrics_xgb  = evaluate_model(model_xgb,  X_test_s, y_test)
    print("📊 Evaluating LightGBM …")
    metrics_lgbm = evaluate_model(model_lgbm, X_test_s, y_test)

    # ── inference latency ────────────────────────────────────────────────────
    print("⏱  Measuring inference latency …")
    latency_xgb  = measure_inference_latency(model_xgb,  X_test_s)
    latency_lgbm = measure_inference_latency(model_lgbm, X_test_s)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("🔍 Computing SHAP values (XGBoost) …")
    _, ranked_xgb  = compute_shap(model_xgb,  X_test_s, feature_names)
    print("🔍 Computing SHAP values (LightGBM) …")
    _, ranked_lgbm = compute_shap(model_lgbm, X_test_s, feature_names)

    # ── prediction stability ──────────────────────────────────────────────────
    stability = compare_predictions(
        metrics_xgb["proba"],  metrics_lgbm["proba"],
        metrics_xgb["pred"],   metrics_lgbm["pred"],
    )

    # ── print results ─────────────────────────────────────────────────────────
    print_shap_comparison(ranked_xgb, ranked_lgbm)

    summary_df = print_comparison_table(
        metrics_xgb, metrics_lgbm,
        latency_xgb, latency_lgbm,
        train_time_xgb, train_time_lgbm,
        stability,
    )

    # ── plots ─────────────────────────────────────────────────────────────────
    if not no_plots and _MPL_AVAILABLE:
        plot_roc_curves(
            y_test,
            metrics_xgb["proba"], metrics_lgbm["proba"],
            metrics_xgb["roc_auc"], metrics_lgbm["roc_auc"],
            save_path=os.path.join(MODEL_DIR, "roc_curves.png"),
        )
        plot_feature_importance(
            ranked_xgb, ranked_lgbm,
            save_path=os.path.join(MODEL_DIR, "feature_importance.png"),
        )
    elif not no_plots and not _MPL_AVAILABLE:
        print("ℹ️  matplotlib not installed — skipping plots (pip install matplotlib)")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare XGBoost vs LightGBM on the LoanSense synthetic dataset."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plot files (useful when matplotlib is unavailable).",
    )
    args = parser.parse_args()
    main(no_plots=args.no_plots)
