"""
shap_service.py  –  SHAP Explainability Microservice
Port: 8001
Uses: model/model.pkl + model/shap_explainer.pkl + model/encoder.pkl + model/scaler.pkl
Artifacts are auto-downloaded from Hugging Face on first startup.
"""

import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from utils.preprocess import preprocess_input, FEATURE_COLUMNS, FEATURE_LABELS
from utils.risk import get_verdict

app = FastAPI(
    title="Loan Eligibility SHAP Service",
    description="SHAP-based explainability for loan eligibility decisions",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

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
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    encoder   = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print("✅ All artifacts loaded")
except FileNotFoundError as e:
    raise RuntimeError(f"Artifact not found after download attempt: {e}")

# Since there's no OHE pipeline, feature names stay as-is after encoding
TRANSFORMED_FEATURE_NAMES = FEATURE_COLUMNS


# ── Schemas ────────────────────────────────────────────────────────────────────
class LoanInput(BaseModel):
    age:                      float = Field(..., ge=18, le=75)
    employment_type:          str
    city_tier:                str
    has_coapplicant:          int   = 0
    monthly_income:           float = Field(..., gt=0)
    credit_score:             float = Field(..., ge=300, le=900)
    total_existing_emi:       float = 0
    requested_loan_amount:    float = Field(..., gt=0)
    loan_tenure_months:       float = Field(..., gt=0)
    work_experience_years:    float = 0
    loan_type:                Optional[str] = None
    marital_status:           Optional[str] = None
    gender:                   Optional[str] = None
    residential_status:       Optional[str] = None
    payment_history:          Optional[str] = None
    dependents:               Optional[int] = None


class FeatureContribution(BaseModel):
    feature:    str
    label:      str
    shap_value: float
    direction:  str   # "positive" | "negative" | "neutral"
    magnitude:  str   # "high" | "medium" | "low"


class ExplanationResponse(BaseModel):
    verdict:           str
    summary:           str
    top_positive:      List[FeatureContribution]
    top_negative:      List[FeatureContribution]
    all_contributions: List[FeatureContribution]
    base_value:        float


# ── Helpers ────────────────────────────────────────────────────────────────────
def _magnitude(val: float, max_abs: float) -> str:
    if max_abs == 0: return "low"
    r = abs(val) / max_abs
    return "high" if r >= 0.6 else "medium" if r >= 0.25 else "low"

def _contribution(feature: str, shap_val: float, max_abs: float) -> FeatureContribution:
    return FeatureContribution(
        feature    = feature,
        label      = FEATURE_LABELS.get(feature, feature),
        shap_value = round(float(shap_val), 4),
        direction  = "positive" if shap_val > 0.001 else "negative" if shap_val < -0.001 else "neutral",
        magnitude  = _magnitude(shap_val, max_abs),
    )

def _summary(verdict: str, pos: list, neg: list) -> str:
    pl = [c.label for c in pos[:2]]
    nl = [c.label for c in neg[:2]]
    if verdict == "Approved":
        return ("Your application looks strong. "
                + (f"Key strengths: {' and '.join(pl)}. " if pl else "")
                + (f"Watch out for: {' and '.join(nl)}." if nl else ""))
    elif verdict == "Pending":
        return ("Your application is borderline. "
                + (f"Strengths: {' and '.join(pl)}. " if pl else "")
                + (f"Areas of concern: {' and '.join(nl)}." if nl else ""))
    else:
        return ("Your application did not meet the criteria. "
                + (f"Main concerns: {' and '.join(nl)}. " if nl else "")
                + (f"Positive factors: {' and '.join(pl)}." if pl else ""))


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "shap-service", "port": 8001}

@app.post("/explain", response_model=ExplanationResponse)
def explain(data: LoanInput):
    try:
        raw      = data.dict()
        df       = preprocess_input(raw)
        df_enc   = encoder.transform_df(df)        # str → int
        X_scaled = scaler.transform(df_enc)        # scale all features

        # SHAP values on scaled input
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1][0])
        else:
            sv = np.array(shap_values[0])

        base_value = float(
            explainer.expected_value
            if not isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value[1]
        )

        # Map SHAP values to feature names (no OHE expansion, 1:1 mapping)
        collapsed: dict = {}
        for fname, sval in zip(TRANSFORMED_FEATURE_NAMES, sv):
            collapsed[fname] = collapsed.get(fname, 0.0) + float(sval)

        prob    = float(pipeline.predict_proba(X_scaled)[0][1])
        score   = round(prob * 100, 2)
        verdict = get_verdict(score)

        max_abs  = max(abs(v) for v in collapsed.values()) or 1.0
        contribs = [_contribution(k, v, max_abs) for k, v in collapsed.items()]
        contribs.sort(key=lambda c: abs(c.shap_value), reverse=True)

        top_pos = [c for c in contribs if c.direction == "positive"][:5]
        top_neg = [c for c in contribs if c.direction == "negative"][:5]

        return ExplanationResponse(
            verdict           = verdict,
            summary           = _summary(verdict, top_pos, top_neg),
            top_positive      = top_pos,
            top_negative      = top_neg,
            all_contributions = contribs,
            base_value        = round(base_value, 4),
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
