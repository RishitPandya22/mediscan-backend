from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import numpy as np
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

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

TIPS = {
    "diabetes": {
        "HIGH": [
            "Reduce sugar & refined carbohydrate intake immediately",
            "Aim for 30 minutes of exercise daily",
            "Stay well hydrated — drink 8+ glasses of water",
            "Schedule a fasting blood glucose test immediately",
            "Work towards a healthy BMI (18.5-24.9)"
        ],
        "LOW": [
            "Maintain a balanced diet rich in vegetables",
            "Keep up regular physical activity",
            "Ensure 7-8 hours of quality sleep",
            "Annual check-ups are still recommended",
            "Avoid smoking and limit alcohol consumption"
        ]
    },
    "heart": {
        "HIGH": [
            "Consult a cardiologist as soon as possible",
            "Reduce sodium intake to lower blood pressure",
            "Stop smoking — it doubles heart disease risk",
            "Begin a medically supervised exercise program",
            "Discuss cholesterol medication with your doctor"
        ],
        "LOW": [
            "Eat heart-healthy foods — avocado, nuts, fish",
            "Cardio exercise 3-5 times per week",
            "Manage stress through meditation or yoga",
            "Check blood pressure and cholesterol annually",
            "Limit alcohol to recommended guidelines"
        ]
    },
    "parkinsons": {
        "HIGH": [
            "Consult a neurologist as soon as possible",
            "Exercise has been shown to slow progression",
            "Consider speech therapy for vocal symptoms",
            "Discuss Levodopa therapy options with your doctor",
            "Join a Parkinson's support group for guidance"
        ],
        "LOW": [
            "Keep your brain active — puzzles, reading, learning",
            "Regular aerobic exercise protects neurological health",
            "Mediterranean diet supports brain health",
            "Prioritise quality sleep for neural repair",
            "Annual neurological check-ups after age 50"
        ]
    }
}

# ─────────────────────────────────────────────
# HELPER — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def get_top_features(model, feature_names, top_n=5):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]) * 100, 2)}
        for i in indices
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

class PDFReportInput(BaseModel):
    username: str
    disease: str
    risk: str
    probability: float
    top_features: list
    whatif: dict
    input_data: dict

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
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW",
        "top_features": get_top_features(diabetes_model, DIABETES_FEATURES)
    }

@app.post("/whatif/diabetes")
def whatif_diabetes(data: DiabetesInput):
    base = [data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree, data.age]
    scenarios = {
        "Lower Glucose by 20": base[:1] + [max(0, data.glucose-20)] + base[2:],
        "Reduce BMI by 5": base[:5] + [max(0, data.bmi-5)] + base[6:],
        "Lower BP by 10": base[:2] + [max(0, data.blood_pressure-10)] + base[3:],
    }
    results = {}
    for s, inp in scenarios.items():
        scaled = diabetes_scaler.transform([inp])
        results[s] = round(float(diabetes_model.predict_proba(scaled)[0][1]) * 100, 2)
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
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW",
        "top_features": get_top_features(heart_model, HEART_FEATURES)
    }

@app.post("/whatif/heart")
def whatif_heart(data: HeartInput):
    base = [data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal]
    scenarios = {
        "Lower Cholesterol by 50": base[:4] + [max(0, data.chol-50)] + base[5:],
        "Reduce BP by 15": base[:3] + [max(0, data.trestbps-15)] + base[4:],
        "Increase Max Heart Rate by 20": base[:7] + [min(220, data.thalach+20)] + base[8:],
    }
    results = {}
    for s, inp in scenarios.items():
        scaled = heart_scaler.transform([inp])
        results[s] = round(float(heart_model.predict_proba(scaled)[0][1]) * 100, 2)
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
    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk": "HIGH" if prediction == 1 else "LOW",
        "top_features": get_top_features(parkinsons_model, PARKINSONS_FEATURES)
    }

@app.post("/whatif/parkinsons")
def whatif_parkinsons(data: ParkinsonsInput):
    base = [data.fo, data.fhi, data.flo, data.jitter_pct, data.jitter_abs,
            data.rap, data.ppq, data.ddp, data.shimmer, data.shimmer_db,
            data.apq3, data.apq5, data.apq, data.dda, data.nhr, data.hnr,
            data.rpde, data.dfa, data.spread1, data.spread2, data.d2, data.ppe]
    scenarios = {
        "Improve HNR by 5": base[:15] + [min(40, data.hnr+5)] + base[16:],
        "Reduce Jitter by 50%": base[:3] + [data.jitter_pct*0.5] + base[4:],
        "Reduce NHR by 50%": base[:14] + [data.nhr*0.5] + base[15:],
    }
    results = {}
    for s, inp in scenarios.items():
        scaled = parkinsons_scaler.transform([inp])
        results[s] = round(float(parkinsons_model.predict_proba(scaled)[0][1]) * 100, 2)
    return {"whatif": results}

# ══════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════
@app.post("/generate-report")
def generate_report(data: PDFReportInput):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )

    # Colors
    dark_bg = colors.HexColor('#020b18')
    neon_green = colors.HexColor('#00ff95')
    neon_blue = colors.HexColor('#00b4ff')
    danger_red = colors.HexColor('#ff4444')
    text_color = colors.HexColor('#e6edf3')
    muted_color = colors.HexColor('#8b949e')
    card_bg = colors.HexColor('#0d1f2d')

    risk_color = danger_red if data.risk == "HIGH" else neon_green

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title',
        fontName='Helvetica-Bold', fontSize=28,
        textColor=neon_green, alignment=TA_CENTER,
        spaceAfter=4, letterSpacing=6
    )
    subtitle_style = ParagraphStyle('subtitle',
        fontName='Helvetica', fontSize=9,
        textColor=neon_blue, alignment=TA_CENTER,
        spaceAfter=2, letterSpacing=3
    )
    section_style = ParagraphStyle('section',
        fontName='Helvetica-Bold', fontSize=10,
        textColor=neon_green, spaceAfter=8,
        letterSpacing=2
    )
    body_style = ParagraphStyle('body',
        fontName='Helvetica', fontSize=9,
        textColor=text_color, spaceAfter=4, leading=14
    )
    muted_style = ParagraphStyle('muted',
        fontName='Helvetica', fontSize=8,
        textColor=muted_color, spaceAfter=4
    )
    risk_style = ParagraphStyle('risk',
        fontName='Helvetica-Bold', fontSize=32,
        textColor=risk_color, alignment=TA_CENTER,
        spaceAfter=4, letterSpacing=4
    )
    disclaimer_style = ParagraphStyle('disclaimer',
        fontName='Helvetica-Oblique', fontSize=7,
        textColor=muted_color, alignment=TA_CENTER,
        spaceAfter=4
    )

    disease_names = {
        'diabetes': 'DIABETES',
        'heart': 'HEART DISEASE',
        'parkinsons': "PARKINSON'S DISEASE"
    }
    disease_icons = {
        'diabetes': '🩸',
        'heart': '🫀',
        'parkinsons': '🫁'
    }

    story = []

    # ── HEADER ──
    story.append(Paragraph("⚕ MEDISCAN AI", title_style))
    story.append(Paragraph("MEDICAL RISK ASSESSMENT REPORT", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", muted_style))
    story.append(HRFlowable(width="100%", thickness=1, color=neon_green, spaceAfter=16))

    # ── PATIENT INFO ──
    story.append(Paragraph("◈ PATIENT INFORMATION", section_style))
    patient_data = [
        ['Patient', data.username],
        ['Assessment Type', f"{disease_icons.get(data.disease, '')} {disease_names.get(data.disease, data.disease.upper())}"],
        ['Report Date', datetime.now().strftime('%B %d, %Y')],
        ['Report ID', f"MSR-{datetime.now().strftime('%Y%m%d%H%M%S')}"],
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_bg),
        ('TEXTCOLOR', (0,0), (0,-1), neon_green),
        ('TEXTCOLOR', (1,0), (1,-1), text_color),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [card_bg, colors.HexColor('#0a1628')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#00ff9520')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 16))

    # ── RISK RESULT ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#00ff9530'), spaceAfter=12))
    story.append(Paragraph("◈ RISK ASSESSMENT RESULT", section_style))
    story.append(Paragraph(
        f"{'⚠ HIGH RISK' if data.risk == 'HIGH' else '✅ LOW RISK'}",
        risk_style
    ))
    story.append(Paragraph(
        f"Risk Probability: {data.probability}%",
        ParagraphStyle('prob', fontName='Helvetica-Bold', fontSize=14,
                       textColor=risk_color, alignment=TA_CENTER, spaceAfter=8)
    ))
    story.append(Paragraph(
        "Please consult a qualified healthcare provider for proper diagnosis and treatment." if data.risk == "HIGH"
        else "No significant risk detected. Maintain your healthy lifestyle!",
        ParagraphStyle('risksub', fontName='Helvetica', fontSize=9,
                       textColor=muted_color, alignment=TA_CENTER, spaceAfter=4)
    ))
    story.append(Spacer(1, 16))

    # ── FEATURE IMPORTANCE ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#00ff9530'), spaceAfter=12))
    story.append(Paragraph("◈ TOP RISK FACTORS", section_style))
    feature_data = [['Factor', 'Importance Score', 'Impact']]
    for f in data.top_features:
        bar = '█' * int(f['importance'] / 5)
        feature_data.append([f['feature'], f"{f['importance']}%", bar])
    feature_table = Table(feature_data, colWidths=[2.5*inch, 1.5*inch, 2.5*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0a1628')),
        ('TEXTCOLOR', (0,0), (-1,0), neon_green),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,1), (-1,-1), text_color),
        ('TEXTCOLOR', (2,1), (2,-1), neon_blue),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [card_bg, colors.HexColor('#0a1628')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#00ff9520')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(feature_table)
    story.append(Spacer(1, 16))

    # ── WHAT-IF SIMULATOR ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#00ff9530'), spaceAfter=12))
    story.append(Paragraph("◈ WHAT-IF HEALTH SIMULATION", section_style))
    story.append(Paragraph(
        "The following scenarios show how improving your health metrics could reduce your risk:",
        body_style
    ))
    whatif_data = [['Scenario', 'Projected Risk', 'Change']]
    for scenario, prob in data.whatif.items():
        change = prob - data.probability
        change_str = f"{'↓' if change < 0 else '↑'} {abs(round(change, 2))}%"
        whatif_data.append([scenario, f"{prob}%", change_str])
    whatif_table = Table(whatif_data, colWidths=[3.5*inch, 1.5*inch, 1.5*inch])
    whatif_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0a1628')),
        ('TEXTCOLOR', (0,0), (-1,0), neon_blue),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,1), (-1,-1), text_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [card_bg, colors.HexColor('#0a1628')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#00b4ff20')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(whatif_table)
    story.append(Spacer(1, 16))

    # ── HEALTH TIPS ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#00ff9530'), spaceAfter=12))
    story.append(Paragraph("◈ HEALTH RECOMMENDATIONS", section_style))
    tips = TIPS.get(data.disease, {}).get(data.risk, [])
    tips_data = [[f"{i+1}. {tip}"] for i, tip in enumerate(tips)]
    tips_table = Table(tips_data, colWidths=[6.5*inch])
    tips_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0,0), (-1,-1), text_color),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [card_bg, colors.HexColor('#0a1628')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#00ff9520')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(tips_table)
    story.append(Spacer(1, 24))

    # ── FOOTER ──
    story.append(HRFlowable(width="100%", thickness=1, color=neon_green, spaceAfter=8))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by MediScan AI for educational purposes only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider.",
        disclaimer_style
    ))
    story.append(Paragraph(
        "MediScan AI — Built by Rishit Pandya | M.Data Science @ University of Adelaide 🇦🇺",
        ParagraphStyle('footer', fontName='Helvetica', fontSize=7,
                       textColor=neon_green, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)

    disease_label = disease_names.get(data.disease, data.disease).replace(' ', '_')
    filename = f"MediScan_Report_{disease_label}_{datetime.now().strftime('%Y%m%d')}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )