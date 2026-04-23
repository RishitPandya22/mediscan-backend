# 🔧 MediScan AI — Backend API v3.0

![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF-red?style=for-the-badge)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

> Production FastAPI backend powering MediScan AI v3.0 — serving 3 trained ML models with disease risk prediction, 95% confidence intervals, feature importance analysis, what-if health simulation, and PDF medical report generation.

🔧 **Live API:** [mediscan-backend-lmhf.onrender.com](https://mediscan-backend-lmhf.onrender.com)
🌐 **Frontend App:** [mediscan-frontend-ruby.vercel.app](https://mediscan-frontend-ruby.vercel.app)
📖 **Interactive Docs:** [mediscan-backend-lmhf.onrender.com/docs](https://mediscan-backend-lmhf.onrender.com/docs)

---

## 🎯 What Does This Do?

This is the Python backend for MediScan AI. It:
- Loads 3 pre-trained scikit-learn ML models on startup
- Exposes REST API endpoints for disease risk prediction
- Returns **95% confidence intervals** using ensemble tree predictions
- Returns **feature importance scores** showing which inputs drove the prediction
- Simulates **what-if health improvement scenarios**
- Generates **professional PDF medical reports** using ReportLab
- Handles CORS so the React frontend can communicate freely

---

## 🚀 API Endpoints

### Prediction Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict/diabetes` | Diabetes risk + CI + feature importance |
| POST | `/predict/heart` | Heart disease risk + CI + feature importance |
| POST | `/predict/parkinsons` | Parkinson's risk + CI + feature importance |

### What-If Simulation Endpoints
| Method | Endpoint | Description |
|---|---|---|
| POST | `/whatif/diabetes` | Simulate diabetes risk improvements |
| POST | `/whatif/heart` | Simulate heart disease risk improvements |
| POST | `/whatif/parkinsons` | Simulate Parkinson's risk improvements |

### Report Endpoints
| Method | Endpoint | Description |
|---|---|---|
| POST | `/generate-report` | Generate downloadable PDF medical report |

---

## 📥 Request & Response Examples

### 🩸 Diabetes Prediction
**POST** `/predict/diabetes`
```json
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 50
}
```
**Response:**
```json
{
  "prediction": 1,
  "probability": 78.45,
  "risk": "HIGH",
  "top_features": [
    {"feature": "Glucose", "importance": 28.5},
    {"feature": "BMI", "importance": 18.2},
    {"feature": "Age", "importance": 14.1},
    {"feature": "Diabetes Pedigree", "importance": 12.3},
    {"feature": "Insulin", "importance": 9.8}
  ],
  "confidence_interval": {
    "lower": 73.2,
    "upper": 83.7,
    "std": 2.6
  }
}
```

### 🔮 What-If Simulation
**POST** `/whatif/diabetes` — same input as prediction.

**Response:**
```json
{
  "whatif": {
    "Lower Glucose by 20": 52.3,
    "Reduce BMI by 5": 61.1,
    "Lower BP by 10": 74.2
  }
}
```

---

## 🧠 ML Models

| Disease | Algorithm | Dataset | Accuracy |
|---|---|---|---|
| 🩸 Diabetes | Random Forest (100 trees) | Pima Indians Diabetes | ~73% |
| 🫀 Heart Disease | Gradient Boosting (100 trees) | Cleveland Heart Disease | ~93% |
| 🫁 Parkinson's | Gradient Boosting (100 trees) | UCI Parkinson's Voice | ~95% |

### Confidence Interval Method
Confidence intervals are calculated using individual tree predictions from the ensemble:
- Each tree in the forest/boosting ensemble makes its own probability prediction
- Standard deviation across all trees is calculated
- 95% CI = probability ± (1.96 × std deviation)
- Falls back to ±5% range if ensemble extraction fails

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| scikit-learn | ML model training & inference |
| joblib | Model serialization |
| pandas + numpy | Data processing |
| ReportLab | PDF report generation |
| uvicorn | ASGI server |
| Render | Cloud deployment (free tier) |

---

## 📁 Project Structure
mediscan-backend/
├── models/
│   ├── diabetes_model.pkl       ← Trained Random Forest
│   ├── diabetes_scaler.pkl      ← StandardScaler
│   ├── heart_model.pkl          ← Trained Gradient Boosting
│   ├── heart_scaler.pkl         ← StandardScaler
│   ├── parkinsons_model.pkl     ← Trained Gradient Boosting
│   └── parkinsons_scaler.pkl    ← StandardScaler
├── main.py                      ← FastAPI app + all endpoints
├── train_models.py              ← Model training script
├── requirements.txt             ← Python dependencies
└── README.md

---

## 🏃 Run Locally

```bash
git clone https://github.com/RishitPandya22/mediscan-backend
cd mediscan-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API live at `http://127.0.0.1:8000` ✅

Visit `http://127.0.0.1:8000/docs` for the **interactive Swagger UI** where you can test all endpoints! 🎯

---

## 💼 Interview Talking Points

> *"The backend is a FastAPI Python application serving 3 scikit-learn ensemble models. Each prediction endpoint returns the risk level, probability score, a 95% confidence interval calculated from individual tree predictions across the ensemble, and top 5 feature importances. There are also what-if simulation endpoints that modify input values to show projected risk changes, and a PDF generation endpoint using ReportLab that produces professional medical-style reports."*

---

## 👨‍💻 About the Developer

**Rishit Pandya**
Master of Data Science Student @ University of Adelaide, South Australia 🇦🇺

[![GitHub](https://img.shields.io/badge/GitHub-RishitPandya22-181717?style=for-the-badge&logo=github)](https://github.com/RishitPandya22)

---

## ⚠️ Disclaimer

This API is built for **educational and portfolio purposes only**. Predictions are NOT medical advice. Always consult a qualified healthcare provider.

---

*Built with 🔥 by Rishit Pandya — M.Data Science @ University of Adelaide*