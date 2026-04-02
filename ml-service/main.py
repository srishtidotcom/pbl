"""
main.py  –  ML Prediction Microservice
Port: 8000
Artifacts (auto-downloaded from Hugging Face on first startup):
  model/model.pkl          – XGBoost classifier
  model/encoder.pkl        – CategoricalEncoder (str → int)
  model/scaler.pkl         – StandardScaler
  model/shap_explainer.pkl – SHAP TreeExplainer
"""

import os, json
import urllib.request
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from utils.preprocess import preprocess_input, FEATURE_COLUMNS
from utils.risk import get_risk_category, get_verdict, get_confidence_label

app = FastAPI(
    title="Loan Eligibility ML Service",
    description="XGBoost pipeline-based loan eligibility prediction",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Artifact registry ──────────────────────────────────────────────────────────
HF_BASE = "https://huggingface.co/Shravni123/loansense-artifacts/resolve/main"

ARTIFACTS: dict[str, str] = {
    "model.pkl":          f"{HF_BASE}/model.pkl",
    "encoder.pkl":        f"{HF_BASE}/encoder.pkl",
    "scaler.pkl":         f"{HF_BASE}/scaler.pkl",
    "shap_explainer.pkl": f"{HF_BASE}/shap_explainer.pkl",
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Download any missing artifacts ────────────────────────────────────────────
for filename, url in ARTIFACTS.items():
    dest = os.path.join(MODEL_DIR, filename)
    if os.path.exists(dest):
        print(f"✅ {filename} already present, skipping download.")
        continue
    print(f"⬇️  Downloading {filename} from Hugging Face …")
    urllib.request.urlretrieve(url, dest)
    if not os.path.exists(dest):
        raise RuntimeError(
            f"Download failed for '{filename}'. "
            f"Check that {url} is publicly accessible."
        )
    print(f"✅ {filename} saved → {dest}")

# ── Load artifacts ─────────────────────────────────────────────────────────────
try:
    pipeline  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    encoder   = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    print("✅ All artifacts loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Artifact not found after download attempt: {e}")

# ── Load optional meta.json ────────────────────────────────────────────────────
meta_path = os.path.join(MODEL_DIR, "meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"version": "1.0.0"}


# ── Schemas ────────────────────────────────────────────────────────────────────
class LoanInput(BaseModel):
    # Personal
    age:                   float = Field(..., ge=18, le=75,   example=30)
    employment_type:       str   = Field(...,                 example="Salaried")
    city_tier:             str   = Field(...,                 example="Metro")
    has_coapplicant:       int   = Field(0,  ge=0, le=1,      example=0)

    # Financial
    monthly_income:        float = Field(..., gt=0,           example=50000)
    credit_score:          float = Field(..., ge=300, le=900, example=720)
    total_existing_emi:    float = Field(0,  ge=0,            example=5000)
    requested_loan_amount: float = Field(..., gt=0,           example=500000)
    loan_tenure_months:    float = Field(..., gt=0,           example=60)

    # Work
    work_experience_years: float = Field(0,  ge=0,            example=5)

    # Optional extras
    loan_type:             Optional[str] = None
    marital_status:        Optional[str] = None
    gender:                Optional[str] = None
    residential_status:    Optional[str] = None
    payment_history:       Optional[str] = None
    dependents:            Optional[int] = None


class PredictionResponse(BaseModel):
    probability:   float
    score:         float   # eligibility score  (0–100, higher = more eligible)
    risk_score:    float   # risk score          (0–100, higher = more risky)
    risk_category: str
    verdict:       str
    confidence:    str
    model_version: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-service", "port": 8000}


@app.get("/model-info")
def model_info():
    return meta


@app.post("/predict", response_model=PredictionResponse)
def predict(data: LoanInput):
    try:
        df       = preprocess_input(data.dict())       # raw DataFrame
        df_enc   = encoder.transform(df)               # categorical → numeric
        X_scaled = scaler.transform(df_enc)            # scale numerics
        prob     = float(pipeline.predict_proba(X_scaled)[0][1])

        score      = round(prob * 100, 2)              # high prob → eligible
        risk_score = round((1 - prob) * 100, 2)        # high prob → low risk

        return PredictionResponse(
            probability   = round(prob, 4),
            score         = score,
            risk_score    = risk_score,
            risk_category = get_risk_category(risk_score),
            verdict       = get_verdict(score),
            confidence    = get_confidence_label(prob),
            model_version = meta.get("version", "1.0.0"),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
