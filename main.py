from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="MediScan AI Backend", version="2.0.0")

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
diabetes_model    = joblib.load("models/diabetes_model.pkl")
diabetes_scaler   = joblib.load("models/diabetes_scaler.pkl")
heart_model       = joblib.load("models/heart_model.pkl")
heart_scaler      = joblib.load("models/heart_scaler.pkl")
parkinsons_model  = joblib.load("models/parkinsons_model.pkl")
parkinsons_scaler = joblib.load("models/parkinsons_scaler.pkl")

# ─────────────────────────────────────────────
# FEATURE NAMES
# ─────────────────────────────────────────────
DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "Blood Pressure",
    "Skin Thickness", "Insulin", "BMI",
    "Diabetes Pedigree", "Age"
]

HEART_FEATURES = [
    "Age", "Sex", "Chest Pain Type", "Resting BP",
    "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
    "Max Heart Rate", "Exercise Angina", "ST Depression",
    "Slope", "Major Vessels", "Thalassemia"
]

PARKINSONS_FEATURES = [
    "MDVP:Fo", "MDVP:Fhi", "MDVP:Flo", "Jitter(%)",
    "Jitter(Abs)", "RAP", "PPQ", "DDP", "Shimmer",
    "Shimmer(dB)", "APQ3", "APQ5", "APQ", "DDA",
    "NHR", "HNR", "RPDE", "DFA", "Spread1",
    "Spread2", "D2", "PPE"
]

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
# HELPER — GET FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def get_top_features(model, feature_names, top_n=5):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]) * 100, 2)}
        for i in indices
    ]

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "MediScan AI Backend v2.0 is running! 🏥"}

# ══════════════════════════════════════════════
# DIABETES
# ══════════════════════════════════════════════
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
    top_features = get_top_features(diabetes_model, DIABETES_FEATURES)
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW",
        "top_features": top_features
    }

@app.post("/whatif/diabetes")
def whatif_diabetes(data: DiabetesInput):
    results = {}
    base_input = [
        data.pregnancies, data.glucose, data.blood_pressure,
        data.skin_thickness, data.insulin, data.bmi,
        data.diabetes_pedigree, data.age
    ]
    # Simulate improvements
    scenarios = {
        "Lower Glucose by 20": base_input[:1] + [max(0, data.glucose - 20)] + base_input[2:],
        "Reduce BMI by 5": base_input[:5] + [max(0, data.bmi - 5)] + base_input[6:],
        "Increase Exercise (lower BP by 10)": base_input[:2] + [max(0, data.blood_pressure - 10)] + base_input[3:],
    }
    for scenario, inputs in scenarios.items():
        scaled = diabetes_scaler.transform([inputs])
        prob = float(diabetes_model.predict_proba(scaled)[0][1])
        results[scenario] = round(prob * 100, 2)
    return {"whatif": results}

# ══════════════════════════════════════════════
# HEART
# ══════════════════════════════════════════════
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
    top_features = get_top_features(heart_model, HEART_FEATURES)
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW",
        "top_features": top_features
    }

@app.post("/whatif/heart")
def whatif_heart(data: HeartInput):
    results = {}
    base_input = [
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]
    scenarios = {
        "Lower Cholesterol by 50": base_input[:4] + [max(0, data.chol - 50)] + base_input[5:],
        "Reduce BP by 15": base_input[:3] + [max(0, data.trestbps - 15)] + base_input[4:],
        "Increase Max Heart Rate by 20": base_input[:7] + [min(220, data.thalach + 20)] + base_input[8:],
    }
    for scenario, inputs in scenarios.items():
        scaled = heart_scaler.transform([inputs])
        prob = float(heart_model.predict_proba(scaled)[0][1])
        results[scenario] = round(prob * 100, 2)
    return {"whatif": results}

# ══════════════════════════════════════════════
# PARKINSONS
# ══════════════════════════════════════════════
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
    top_features = get_top_features(parkinsons_model, PARKINSONS_FEATURES)
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW",
        "top_features": top_features
    }

@app.post("/whatif/parkinsons")
def whatif_parkinsons(data: ParkinsonsInput):
    results = {}
    base_input = [
        data.fo, data.fhi, data.flo, data.jitter_pct, data.jitter_abs,
        data.rap, data.ppq, data.ddp, data.shimmer, data.shimmer_db,
        data.apq3, data.apq5, data.apq, data.dda, data.nhr, data.hnr,
        data.rpde, data.dfa, data.spread1, data.spread2, data.d2, data.ppe
    ]
    scenarios = {
        "Improve HNR by 5": base_input[:15] + [min(40, data.hnr + 5)] + base_input[16:],
        "Reduce Jitter by 50%": base_input[:3] + [data.jitter_pct * 0.5] + base_input[4:],
        "Reduce NHR by 50%": base_input[:14] + [data.nhr * 0.5] + base_input[15:],
    }
    for scenario, inputs in scenarios.items():
        scaled = parkinsons_scaler.transform([inputs])
        prob = float(parkinsons_model.predict_proba(scaled)[0][1])
        results[scenario] = round(prob * 100, 2)
    return {"whatif": results}