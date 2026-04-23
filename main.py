from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="MediScan AI Backend", version="1.0.0")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
diabetes_model   = joblib.load("models/diabetes_model.pkl")
diabetes_scaler  = joblib.load("models/diabetes_scaler.pkl")
heart_model      = joblib.load("models/heart_model.pkl")
heart_scaler     = joblib.load("models/heart_scaler.pkl")
parkinsons_model = joblib.load("models/parkinsons_model.pkl")
parkinsons_scaler= joblib.load("models/parkinsons_scaler.pkl")

# ─────────────────────────────────────────────
# INPUT SCHEMAS
# ─────────────────────────────────────────────
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class ParkinsonsInput(BaseModel):
    fo: float
    fhi: float
    flo: float
    jitter_pct: float
    jitter_abs: float
    rap: float
    ppq: float
    ddp: float
    shimmer: float
    shimmer_db: float
    apq3: float
    apq5: float
    apq: float
    dda: float
    nhr: float
    hnr: float
    rpde: float
    dfa: float
    spread1: float
    spread2: float
    d2: float
    ppe: float

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "MediScan AI Backend is running! 🏥"}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    input_array = np.array([[
        data.pregnancies, data.glucose, data.blood_pressure,
        data.skin_thickness, data.insulin, data.bmi,
        data.diabetes_pedigree, data.age
    ]])
    scaled = diabetes_scaler.transform(input_array)
    prediction = int(diabetes_model.predict(scaled)[0])
    probability = float(diabetes_model.predict_proba(scaled)[0][1])
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW"
    }

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    input_array = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    scaled = heart_scaler.transform(input_array)
    prediction = int(heart_model.predict(scaled)[0])
    probability = float(heart_model.predict_proba(scaled)[0][1])
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW"
    }

@app.post("/predict/parkinsons")
def predict_parkinsons(data: ParkinsonsInput):
    input_array = np.array([[
        data.fo, data.fhi, data.flo, data.jitter_pct, data.jitter_abs,
        data.rap, data.ppq, data.ddp, data.shimmer, data.shimmer_db,
        data.apq3, data.apq5, data.apq, data.dda, data.nhr, data.hnr,
        data.rpde, data.dfa, data.spread1, data.spread2, data.d2, data.ppe
    ]])
    scaled = parkinsons_scaler.transform(input_array)
    prediction = int(parkinsons_model.predict(scaled)[0])
    probability = float(parkinsons_model.predict_proba(scaled)[0][1])
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW"
    }