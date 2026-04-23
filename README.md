# 🔧 MediScan AI — Backend API

![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

> FastAPI backend powering MediScan AI — serving 3 trained ML models for disease risk prediction via REST API endpoints.

🔧 **Live API:** [mediscan-backend-lmhf.onrender.com](https://mediscan-backend-lmhf.onrender.com)
🌐 **Frontend App:** [mediscan-frontend-ruby.vercel.app](https://mediscan-frontend-ruby.vercel.app)

---

## 🎯 What Does This Do?

This is the Python backend for MediScan AI. It:
- Loads 3 pre-trained scikit-learn ML models on startup
- Exposes REST API endpoints for disease risk prediction
- Accepts patient health data as JSON input
- Returns risk prediction, probability score, and risk level
- Handles CORS so the React frontend can communicate with it

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check — confirms API is running |
| POST | `/predict/diabetes` | Diabetes risk prediction |
| POST | `/predict/heart` | Heart disease risk prediction |
| POST | `/predict/parkinsons` | Parkinson's risk prediction |

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
  "risk": "HIGH"
}
```

### 🫀 Heart Disease Prediction
**POST** `/predict/heart`
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```
**Response:**
```json
{
  "prediction": 1,
  "probability": 91.28,
  "risk": "HIGH"
}
```

---

## 🧠 ML Models

| Disease | Algorithm | Dataset | Accuracy |
|---|---|---|---|
| 🩸 Diabetes | Random Forest (100 trees) | Pima Indians Diabetes | ~73% |
| 🫀 Heart Disease | Gradient Boosting (100 trees) | Cleveland Heart Disease | ~93% |
| 🫁 Parkinson's | Gradient Boosting (100 trees) | UCI Parkinson's Voice | ~95% |

All models are trained with `scikit-learn`, scaled with `StandardScaler`, and saved as `.pkl` files using `joblib`.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| scikit-learn | ML model training & inference |
| joblib | Model serialization |
| pandas + numpy | Data processing |
| uvicorn | ASGI server |
| Render | Cloud deployment |

---

## 📁 Project Structure
mediscan-backend/
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── parkinsons_model.pkl
│   └── parkinsons_scaler.pkl
├── main.py                ← FastAPI app + all endpoints
├── train_models.py        ← Model training script
├── requirements.txt       ← Python dependencies
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

API will be live at `http://127.0.0.1:8000` ✅

Visit `http://127.0.0.1:8000/docs` for the **interactive Swagger UI** where you can test all endpoints! 🎯

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