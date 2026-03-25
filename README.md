# 🏦 LoanSense — AI-Powered Loan Eligibility Advisor

> An intelligent loan eligibility assessment platform combining rule-based decision engines, XGBoost ML scoring, and SHAP explainability — built as a full-stack MERN + Python microservices project.

---

## 📌 What It Does

LoanSense helps users understand their loan eligibility **before** they apply to a bank. Instead of a black-box yes/no, it gives:

- ✅ **Rule-based hard eligibility checks** (FOIR, LTV, credit score thresholds)
- 🤖 **ML-based probability scoring** (XGBoost trained on synthetic financial data)
- 🔍 **SHAP explainability** — "Why was I approved/rejected?" with top contributing factors
- 📊 **Approved loan offer** — max amount, tenure, interest rate, EMI

Supports 5 loan types: Personal, Home, Education, Vehicle, Business.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│              (Vite + Tailwind, port 5556)                │
└───────────────────────┬─────────────────────────────────┘
                        │ REST API
┌───────────────────────▼─────────────────────────────────┐
│              Node.js / Express Backend                   │
│               (MongoDB, port 5000)                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Rule Engine → ML Service → SHAP Service        │    │
│  └──────┬──────────────────┬───────────────────────┘    │
└─────────┼──────────────────┼────────────────────────────┘
          │                  │
┌─────────▼──────┐  ┌────────▼───────┐
│   ML Service   │  │  SHAP Service  │
│  FastAPI :8000 │  │  FastAPI :8001 │
│   XGBoost      │  │ TreeExplainer  │
└────────────────┘  └────────────────┘
```

---

## 🗂️ Project Structure

```
pbl/
├── frontend/                   # React + Vite + Tailwind
│   └── src/
│       ├── pages/
│       │   ├── Landing.jsx
│       │   ├── Login.jsx
│       │   ├── Signup.jsx
│       │   ├── Dashboard.jsx
│       │   ├── FinancialProfile.jsx
│       │   ├── LoanCheck.jsx       ← multi-step loan form
│       │   ├── LoanDetail.jsx      ← results + SHAP viz
│       │   ├── LoanHistory.jsx
│       │   └── Settings.jsx
│       ├── context/AppContext.jsx
│       └── utils/enums.js
│
├── backend/                    # Node.js + Express + MongoDB
│   ├── config/
│   │   ├── mongodb.js
│   │   └── cloudinary.js
│   ├── controllers/
│   │   ├── user_controller.js
│   │   ├── loanEligibilityCheckController.js
│   │   └── financialProfile_controller.js
│   ├── models/
│   │   ├── User.js
│   │   ├── FinancialProfile.js
│   │   └── LoanEligibilityCheck.js
│   ├── routes/
│   │   ├── user_routes.js
│   │   ├── financialProfile_routes.js
│   │   └── loanEligibilityCheck_routes.js
│   ├── services/
│   │   ├── user_service.js
│   │   ├── loanEligibilityService.js  ← rule engine + ML integration
│   │   ├── financialProfile_service.js
│   │   └── mlservice.js               ← calls ML + SHAP microservices
│   ├── validators/
│   ├── middlewares/
│   ├── constants/enums.js
│   └── server.js
│
└── ml-service/                 # Python FastAPI microservices
    ├── model/
    │   ├── train_model.py
    │   ├── model_pipeline.pkl      ← trained XGBoost pipeline
    │   └── shap_explainer.pkl      ← TreeExplainer
    ├── utils/
    │   ├── preprocess.py           ← feature engineering
    │   └── risk.py                 ← score → risk category
    ├── main.py                     ← /predict endpoint (port 8000)
    ├── shap_service.py             ← /explain endpoint (port 8001)
    ├── generate_synthetic_data.py
    └── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Node.js v20+
- Python 3.10+
- MongoDB 7.0+

---

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd pbl
```

---

### 2. Set up the ML service

```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you need to retrain the model:
```bash
python3 generate_synthetic_data.py
python3 model/train_model.py
```

---

### 3. Set up the backend

```bash
cd backend
npm install
```

Create `backend/.env`:
```env
PORT=5000
MONGODB_URL=mongodb://localhost:27017/loan_eligibility
JWT_SECRET=your_secret_key_here
JWT_EXPIRES_IN=7d
FRONTEND_URL=http://localhost:5556
ML_SERVICE_URL=http://localhost:8000
SHAP_SERVICE_URL=http://localhost:8001
```

---

### 4. Set up the frontend

```bash
cd frontend
npm install --legacy-peer-deps
```

Create `frontend/.env`:
```env
VITE_BACKEND_URL=http://localhost:5000
```

---

### 5. Run everything

Open **5 terminals**:

```bash
# Terminal 1 — MongoDB
sudo systemctl start mongod

# Terminal 2 — Backend
cd backend && npm run server

# Terminal 3 — ML Service
cd ml-service && source venv/bin/activate && uvicorn main:app --reload --port 8000

# Terminal 4 — SHAP Service
cd ml-service && source venv/bin/activate && uvicorn shap_service:app --reload --port 8001

# Terminal 5 — Frontend
cd frontend && npm run dev
```

Open **http://localhost:5556** in your browser.

---

## 🔌 API Reference

### Backend (port 5000)

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| POST | `/user/sign-up` | ❌ | Register new user |
| POST | `/user/login` | ❌ | Login, returns JWT |
| GET | `/user/profile` | ✅ | Get user profile |
| POST | `/financial-profile` | ✅ | Create financial profile |
| GET | `/financial-profile` | ✅ | Get financial profile |
| PATCH | `/financial-profile` | ✅ | Update financial profile |
| POST | `/loan-eligibility` | ✅ | Run eligibility check |
| GET | `/loan-eligibility` | ✅ | Get all past checks |
| GET | `/loan-eligibility/:id` | ✅ | Get single check |

### ML Service (port 8000)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/predict` | Returns score, probability, risk, verdict |
| GET | `/health` | Liveness check |
| GET | `/model-info` | Model metadata |

### SHAP Service (port 8001)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/explain` | Returns SHAP feature contributions |
| GET | `/health` | Liveness check |

---

## 🤖 ML Pipeline

### Features used (11 total)

| Feature | Type |
|---------|------|
| age | Numeric |
| monthly_income | Numeric |
| employment_tenure_months | Numeric |
| credit_score | Numeric |
| total_existing_emi | Numeric |
| requested_loan_amount | Numeric |
| dti_ratio (derived) | Numeric |
| loan_to_income (derived) | Numeric |
| has_coapplicant | Numeric (0/1) |
| employment_type | Categorical (OHE) |
| city_tier | Categorical (OHE) |

### Model

- **Algorithm**: XGBoost Classifier
- **Pipeline**: `ColumnTransformer(StandardScaler + OneHotEncoder)` → `XGBClassifier`
- **Explainability**: SHAP `TreeExplainer` — collapses OHE features back to original column names

### Risk Categories

| Score | Risk | Verdict |
|-------|------|---------|
| 80–100 | Very Low | Approved |
| 65–79 | Low | Approved |
| 45–64 | Medium | Pending |
| 25–44 | High | Rejected |
| 0–24 | Very High | Rejected |

---

## 🔒 Rule Engine

Hard rejection rules run **before** ML scoring:

- Credit score < 600 → rejected
- Serious default in payment history → rejected
- FOIR > 50% (total obligations / income) → rejected
- Requested amount > loan type maximum → rejected
- 3+ loan inquiries in last 6 months → rejected

**Loan-type specific rules:**
- Personal Loan: min income ₹25,000, no unemployed/student
- Home Loan: min income ₹35,000, LTV ≤ 80%
- Vehicle Loan: LTV ≤ 85%, vehicle age ≤ 5 years
- Education Loan: co-applicant mandatory, FOIR on co-applicant
- Business Loan: vintage ≥ 12 months, turnover ≥ ₹12L, no GST default

---

## 🛡️ Security

- JWT authentication (HTTP header based)
- bcrypt password hashing
- Zod schema validation on all inputs
- CORS restricted to frontend origin

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite 8, Tailwind CSS 4 |
| Backend | Node.js, Express 5, MongoDB, Mongoose |
| Auth | JWT, bcrypt |
| Validation | Zod |
| ML | XGBoost, scikit-learn, SHAP |
| ML API | FastAPI, Uvicorn |
| File Upload | Cloudinary, Multer |

---

## ⚠️ Disclaimer

> This is an **advisory system**, not a lender. All loan eligibility assessments are based on mock rules and a model trained on synthetic data. Results do not represent actual bank decisions. No real credit bureau data is used.



## ⛓️ Blockchain Audit & Data Integrity (Live)

Every loan eligibility check and its corresponding SHAP explanation data (`summary`, `topPositive`, `topNegative`, `baseValue`) is now anchored to the **Ethereum Sepolia Testnet** via **IPFS**. This ensures that the decision logic is immutable and verifiable.

**The Blockchain Integration provides:**
- **Decentralized Storage:** Full result metadata is pinned to IPFS via Pinata.
- **On-Chain Anchoring:** The IPFS CID is stored in a Solidity smart contract for permanent auditing.
- **Tamper-Proof Verification:** Users receive a unique Transaction Hash per eligibility check to verify their results on Etherscan.

The `mlExplanation` object used for this decentralized anchor looks like:

```json
{
  "summary": "Your application looks strong. Key strengths: Credit Score and Monthly Income.",
  "topPositive": [{ "feature": "credit_score", "label": "Credit Score", "shap_value": 0.42, "direction": "positive", "magnitude": "high" }],
  "topNegative": [{ "feature": "dti_ratio", "label": "Debt-to-Income Ratio", "shap_value": -0.18, "direction": "negative", "magnitude": "medium" }],
  "baseValue": 0.48,
  "blockchainTxHash": "0x621b557a5cd8d839e98dd0062ce1cb172de4bd91f1b84699eb56945ba1088349",
  "ipfsHash": "QmfQh3D4wA5GQwm6bBstHXD2Ygi97C5WQ7yQ2a6Psj5wLB"
}

---

## 👥 Team

Built as part of a Project-Based Learning (PBL) assignment
by Srishti Pandey, Tejas Khadilkar and Shravni Thakur.
