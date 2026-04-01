"""
model/train_model.py
Run from ml-service root:  python3 model/train_model.py
Saves: model.pkl  scaler.pkl  encoder.pkl  meta.json
"""

import sys, os, json
from datetime import datetime
import joblib
import shap          # ← add this line

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import lightgbm as lgb
import joblib

from utils.preprocess import CategoricalEncoder, FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "synthetic_loan_data.csv")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("📂 Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"   Rows: {len(df)} | Approval rate: {df['eligible'].mean():.1%}")

# ── 1b. Reconcile CSV column names → FEATURE_COLUMNS ──────────────────────────
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

# ── 2. Encode categoricals → encoder.pkl ──────────────────────────────────────
print("🔤 Fitting CategoricalEncoder …")
encoder = CategoricalEncoder()
df_enc  = encoder.fit_transform_df(df)

# ── 3. Split X / y ─────────────────────────────────────────────────────────────
X = df_enc[FEATURE_COLUMNS].values
y = df_enc["eligible"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scale → scaler.pkl ──────────────────────────────────────────────────────
print("📐 Fitting StandardScaler …")
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5. Train LightGBM ──────────────────────────────────────────────────────────
print("🚀 Training LightGBM …")
model = LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary', metric='binary_logloss',
    boosting_type='gbdt', num_leaves=31,
    random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)],
          callbacks=[lgb.log_evaluation(period=0)])

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]
acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
cv      = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="roc_auc")

print(f"\n📊 Results  |  Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}  |  CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── 7. Save all artifacts ──────────────────────────────────────────────────────
joblib.dump(model,   os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))

meta = {
    "model_type": "LGBMClassifier", "version": "1.0.0",
    "feature_columns": FEATURE_COLUMNS, "numeric_columns": NUMERIC_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS, "n_features": len(FEATURE_COLUMNS),
    "trained_at": datetime.utcnow().isoformat() + "Z",
    "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
    "accuracy": round(float(acc), 4), "roc_auc": round(float(auc), 4),
    "cv_auc_mean": round(float(cv.mean()), 4), "cv_auc_std": round(float(cv.std()), 4),
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Saved: model/model.pkl  model/scaler.pkl  model/encoder.pkl  model/meta.json")

# Generate and save SHAP explainer
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
print("✅ Saved: shap_explainer.pkl")