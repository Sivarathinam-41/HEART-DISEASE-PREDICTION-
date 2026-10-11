import os
import json
import pickle
import pandas as pd
import numpy as np

# Feature definitions for Cleveland (13 features) and Expanded (11 features)
CLEVELAND_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

EXPANDED_FEATURES = [
    'age', 'resting_bp_s', 'cholesterol', 'max_heart_rate',
    'oldpeak', 'sex', 'chest_pain_type', 'fasting_blood_sugar',
    'resting_ecg', 'exercise_angina', 'st_slope'
]

# Backward compatibility alias
FEATURE_ORDER = CLEVELAND_FEATURES

_EXPANDED_MODEL = None
_CLEVELAND_MODEL = None
_METRICS_CACHE = None

def get_expanded_model():
    """Load the primary 1,190-sample Expanded model pipeline."""
    global _EXPANDED_MODEL
    if _EXPANDED_MODEL is None:
        path = os.path.join(os.path.dirname(__file__), 'heart_disease_expanded_model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}. Run train_expanded.py first.")
        with open(path, 'rb') as f:
            _EXPANDED_MODEL = pickle.load(f)
    return _EXPANDED_MODEL

def get_cleveland_model():
    """Load the 303-sample Cleveland benchmark model pipeline."""
    global _CLEVELAND_MODEL
    if _CLEVELAND_MODEL is None:
        path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                _CLEVELAND_MODEL = pickle.load(f)
    return _CLEVELAND_MODEL

def get_metrics():
    """Return metrics for the active expanded model."""
    global _METRICS_CACHE
    if _METRICS_CACHE is None:
        path = os.path.join(os.path.dirname(__file__), 'model_metrics_expanded.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                _METRICS_CACHE = json.load(f)
        else:
            _METRICS_CACHE = {}
    return _METRICS_CACHE

def explain_risk_factors(data):
    """Generate patient-specific clinical factor insights."""
    factors = []
    bp = data.get('trestbps', data.get('resting_bp_s', 120))
    chol = data.get('chol', data.get('cholesterol', 200))
    cp = data.get('cp', data.get('chest_pain_type', 0))
    exang = data.get('exang', data.get('exercise_angina', 0))
    oldpeak = data.get('oldpeak', 0.0)
    ca = data.get('ca', 0)
    thal = data.get('thal', 1)
    hr = data.get('thalach', data.get('max_heart_rate', 150))
    age = data.get('age', 50)

    if bp >= 140:
        factors.append(f"Elevated resting blood pressure ({int(bp)} mmHg)")
    if chol >= 240:
        factors.append(f"High serum cholesterol ({int(chol)} mg/dl)")
    if cp in [1, 2, 3, 4]:
        factors.append("Reported chest pain indicators present")
    if exang == 1:
        factors.append("Exercise-induced angina detected during screening")
    if oldpeak >= 1.5:
        factors.append(f"Significant ST depression ({oldpeak} mm) indicating ischemic risk")
    if ca >= 1:
        factors.append(f"Fluoroscopy demonstrates {int(ca)} major vessel(s) affected")
    if thal == 3:
        factors.append("Reversible perfusion defect identified")
    if hr < 120 and age < 60:
        factors.append(f"Lower chronotropic reserve / maximum heart rate response ({int(hr)} bpm)")

    if not factors:
        factors.append("All baseline cardiac diagnostic markers within normal screening parameters")
    return factors

def predict_heart_disease(raw_input, model_type='expanded'):
    """
    Primary prediction function using the expanded 1,190-sample model
    with fallback to Cleveland model if requested.
    """
    # Map incoming form fields to expanded schema
    age = float(raw_input.get('age', 50))
    sex = float(raw_input.get('sex', 1))
    cp_val = float(raw_input.get('cp', raw_input.get('chest_pain_type', 0)))
    # Convert 0-indexed cp to 1-indexed for expanded dataset if needed
    cp_expanded = (cp_val + 1) if cp_val in [0, 1, 2, 3] and 'chest_pain_type' not in raw_input else cp_val
    trestbps = float(raw_input.get('trestbps', raw_input.get('resting_bp_s', 125)))
    chol = float(raw_input.get('chol', raw_input.get('cholesterol', 210)))
    fbs = float(raw_input.get('fbs', raw_input.get('fasting_blood_sugar', 0)))
    restecg = float(raw_input.get('restecg', raw_input.get('resting_ecg', 0)))
    thalach = float(raw_input.get('thalach', raw_input.get('max_heart_rate', 155)))
    exang = float(raw_input.get('exang', raw_input.get('exercise_angina', 0)))
    oldpeak = float(raw_input.get('oldpeak', 0.6))
    slope_val = float(raw_input.get('slope', raw_input.get('st_slope', 1)))
    st_slope = (slope_val + 1) if slope_val in [0, 1, 2] and 'st_slope' not in raw_input else slope_val

    if model_type == 'cleveland':
        pipeline = get_cleveland_model()
        ca = float(raw_input.get('ca', 0))
        thal = float(raw_input.get('thal', 2))
        cleveland_dict = {
            'age': age, 'sex': sex, 'cp': cp_val, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope_val,
            'ca': ca, 'thal': thal
        }
        df_in = pd.DataFrame([cleveland_dict], columns=CLEVELAND_FEATURES)
    else:
        # Primary Expanded Model (1,190 samples)
        pipeline = get_expanded_model()
        expanded_dict = {
            'age': age,
            'resting_bp_s': trestbps,
            'cholesterol': chol,
            'max_heart_rate': thalach,
            'oldpeak': oldpeak,
            'sex': sex,
            'chest_pain_type': cp_expanded,
            'fasting_blood_sugar': fbs,
            'resting_ecg': restecg,
            'exercise_angina': exang,
            'st_slope': st_slope
        }
        df_in = pd.DataFrame([expanded_dict], columns=EXPANDED_FEATURES)

    prediction = int(pipeline.predict(df_in)[0])
    prob = float(pipeline.predict_proba(df_in)[0][1])
    risk_percentage = round(prob * 100, 1)

    if risk_percentage < 35.0:
        risk_level = "Low Risk"
        badge_class = "success"
        color = "text-success"
        icon = "❤️"
        summary = "Cardiovascular diagnostic markers indicate low risk of coronary narrowing."
    elif risk_percentage < 65.0:
        risk_level = "Moderate Risk"
        badge_class = "warning"
        color = "text-warning"
        icon = "⚠️"
        summary = "Intermediate cardiovascular markers detected. Clinical follow-up advised."
    else:
        risk_level = "High Risk"
        badge_class = "danger"
        color = "text-danger"
        icon = "🚨"
        summary = "Elevated indicators strongly correlated with coronary artery disease (>50% stenosis)."

    risk_factors = explain_risk_factors(raw_input)

    return {
        'prediction': prediction,
        'probability': prob,
        'risk_percentage': risk_percentage,
        'risk_level': risk_level,
        'badge_class': badge_class,
        'color': color,
        'icon': icon,
        'summary': summary,
        'risk_factors': risk_factors,
        'dataset_used': 'Expanded (1,190 records)' if model_type == 'expanded' else 'Cleveland (303 records)'
    }

def get_demo_presets():
    """Standard pre-filled patient profiles."""
    return {
        'healthy': {
            'label': 'Low-Risk Healthy Profile',
            'data': {
                'age': 38, 'sex': 0, 'cp': 0, 'trestbps': 118,
                'chol': 180, 'fbs': 0, 'restecg': 0, 'thalach': 175,
                'exang': 0, 'oldpeak': 0.2, 'slope': 2, 'ca': 0, 'thal': 1
            }
        },
        'moderate': {
            'label': 'Moderate Borderline Profile',
            'data': {
                'age': 54, 'sex': 1, 'cp': 1, 'trestbps': 135,
                'chol': 235, 'fbs': 0, 'restecg': 1, 'thalach': 148,
                'exang': 0, 'oldpeak': 1.2, 'slope': 1, 'ca': 1, 'thal': 2
            }
        },
        'high_risk': {
            'label': 'High-Risk Cardiac Patient',
            'data': {
                'age': 65, 'sex': 1, 'cp': 3, 'trestbps': 160,
                'chol': 286, 'fbs': 1, 'restecg': 2, 'thalach': 108,
                'exang': 1, 'oldpeak': 3.6, 'slope': 1, 'ca': 2, 'thal': 3
            }
        }
    }

def generate_clinical_recommendations(data, risk_result=None):
    """
    Generate personalized lifestyle, nutritional, and clinical follow-up
    recommendations based on individual physiological biomarkers.
    """
    risk_level = risk_result.get('risk_level', 'Moderate Risk') if risk_result else 'Moderate Risk'
    prob = risk_result.get('risk_percentage', 50.0) if risk_result else 50.0

    bp = float(data.get('trestbps', data.get('resting_bp_s', 125)))
    chol = float(data.get('chol', data.get('cholesterol', 210)))
    oldpeak = float(data.get('oldpeak', 0.6))
    exang = int(float(data.get('exang', data.get('exercise_angina', 0))))
    age = int(float(data.get('age', 52)))
    max_hr = int(float(data.get('thalach', data.get('max_heart_rate', 155))))

    # 1. Nutrition & Diet
    diet = [
        "Adopt a Mediterranean or DASH dietary pattern emphasizing leafy greens, berries, legumes, and whole grains.",
        "Limit saturated fats to under 6% of daily caloric intake and eliminate trans fats to protect endothelial lining."
    ]
    if bp >= 130:
        diet.append(f"Restrict daily sodium intake to under 1,500 mg (your resting BP is {int(bp)} mmHg). Avoid processed foods.")
    if chol >= 200:
        diet.append(f"Incorporate 5–10g of soluble viscous fiber (oats, psyllium, beans) daily to actively reduce serum LDL cholesterol (current: {int(chol)} mg/dL).")

    # 2. Exercise & Physical Conditioning
    max_predicted_hr = 220 - age
    target_aerobic_min = int(max_predicted_hr * 0.50)
    target_aerobic_max = int(max_predicted_hr * 0.70)

    if exang == 1 or oldpeak >= 1.5:
        exercise = [
            "Clinical Precaution: Exercise-induced angina / ST depression detected. Do NOT start intense workouts without physician clearance.",
            "Begin with supervised, low-impact walking (15–20 min at a leisurely pace), monitoring for chest tightness or shortness of breath.",
            "Consult a clinical exercise physiologist or enroll in an outpatient Cardiac Rehabilitation program."
        ]
    else:
        exercise = [
            f"Aim for at least 150 minutes of moderate-intensity aerobic physical activity weekly (e.g. brisk walking, cycling, swimming).",
            f"Target safe aerobic training zone: {target_aerobic_min}–{target_aerobic_max} bpm (50%–70% of estimated maximum heart rate for age {age}).",
            "Incorporate 2 days per week of moderate resistance training to improve insulin sensitivity."
        ]

    # 3. Clinical Biomarker Targets
    biomarkers = [
        {"metric": "Blood Pressure", "current": f"{int(bp)} mmHg", "target": "< 120/80 mmHg", "status": "Elevated" if bp >= 130 else "Normal"},
        {"metric": "Serum Cholesterol", "current": f"{int(chol)} mg/dL", "target": "< 200 mg/dL (LDL < 70 for high risk)", "status": "High" if chol >= 240 else ("Borderline" if chol >= 200 else "Optimal")},
        {"metric": "ST Segment Response", "current": f"{oldpeak} mm", "target": "< 1.0 mm (Normal baseline)", "status": "Ischemia Risk" if oldpeak >= 1.5 else "Stable"}
    ]

    # 4. Medical Follow-up
    if risk_level == "High Risk":
        medical_actions = [
            "🚨 Urgent Cardiology Evaluation: Schedule a formal consultation with a certified cardiologist within 7–14 days.",
            "Diagnostic Workup: Request a 12-lead resting ECG, 2D Transthoracic Echocardiogram, and consideration of a Coronary CT Angiography (CCTA).",
            "Pharmacotherapy Review: Discuss lipid-lowering therapy (statin/ezetimibe) and anti-hypertensive regimen with your physician."
        ]
    elif risk_level == "Moderate Risk":
        medical_actions = [
            "Schedule a primary care cardiovascular follow-up within 30 days.",
            "Request a complete fasting lipid panel, hs-CRP (inflammatory marker), and HbA1c to assess cardiometabolic health.",
            "Consider a Non-Invasive Treadmill Stress Test (TMT) to evaluate functional ischemic response."
        ]
    else:
        medical_actions = [
            "Maintain routine annual cardiovascular screening and annual wellness checkups.",
            "Monitor blood pressure and resting heart rate quarterly.",
            "Continue proactive lifestyle management and heart-healthy dietary habits."
        ]

    return {
        'diet': diet,
        'exercise': exercise,
        'biomarkers': biomarkers,
        'medical_actions': medical_actions,
        'risk_summary': {
            'risk_level': risk_level,
            'probability': prob
        }
    }

def get_cardio_chat_response(message, patient_data=None, risk_result=None):
    """
    Intelligent conversational medical AI advisor that responds to user inquiries
    in the context of cardiovascular health and the patient's specific screening results.
    """
    msg = message.strip().lower()

    # Context variables
    prob = risk_result.get('risk_percentage', None) if risk_result else None
    level = risk_result.get('risk_level', None) if risk_result else None

    # 1. Emergency Symptoms Query
    if any(k in msg for k in ['emergency', 'heart attack symptoms', 'chest pain now', 'severe pain', '911', 'hospital']):
        return (
            "🚨 **CRITICAL MEDICAL WARNING:** If you or someone you are with is currently experiencing crushing chest pressure, "
            "pain radiating to the left arm, jaw, neck, or back, shortness of breath, sudden cold sweats, or dizziness, "
            "**immediately call emergency services (911 / 112) or go to the nearest emergency department.** "
            "Do not attempt to drive yourself. This web application is an educational screening simulator, not an emergency tool."
        )

    # 2. Risk Score & Prediction Questions
    if any(k in msg for k in ['my risk', 'my score', 'result mean', 'why high', 'why low', 'what is my probability']):
        if prob is not None:
            return (
                f"📊 **Your Screening Analysis:**\n\n"
                f"Your estimated cardiovascular disease risk is **{prob}% ({level})**, calculated by our Random Forest pipeline trained on 1,190 multi-center clinical patient records.\n\n"
                f"- **What this means:** Based on your physiological profile, your probability aligns with a **{level}** profile.\n"
                f"- **Next Steps:** Please review the *AI Clinical Suggestions* tab for customized nutritional, exercise, and physician consultation recommendations."
            )
        else:
            return (
                "Please run a cardiovascular risk assessment on the **Screening Form** tab first or select a demo preset. "
                "Once analyzed, I will break down your exact probability score and personalized clinical drivers!"
            )

    # 3. ST Depression (Oldpeak) Questions
    if any(k in msg for k in ['oldpeak', 'st depression', 'st segment', 'depression']):
        return (
            "📉 **What is ST Depression (Oldpeak)?**\n\n"
            "During an electrocardiogram (ECG) stress test, the ST segment represents the interval between ventricular depolarization and repolarization.\n\n"
            "• **Clinical Meaning:** When the ST segment drops (depresses) relative to baseline during exercise, it is a hallmark sign of **myocardial ischemia**—meaning heart muscle is receiving insufficient oxygenated blood due to narrowed coronary arteries.\n"
            "• **Standard Interpretation:**\n"
            "  - `0.0 to 1.0 mm`: Normal / Minimal ischemia.\n"
            "  - `1.0 to 2.0 mm`: Moderate ischemia.\n"
            "  - `> 2.0 mm`: Severe exercise-induced myocardial ischemia requiring prompt cardiologist evaluation."
        )

    # 4. Blood Pressure Questions
    if any(k in msg for k in ['blood pressure', 'bp', 'trestbps', 'hypertension', '140', '120']):
        return (
            "❤️ **Resting Blood Pressure (trestbps) & Heart Health:**\n\n"
            "Blood pressure measures the force exerted against artery walls as your heart pumps.\n\n"
            "• **Clinical Categories (AHA/ACC):**\n"
            "  - **Normal:** < 120/80 mmHg\n"
            "  - **Elevated:** 120–129 / < 80 mmHg\n"
            "  - **Stage 1 Hypertension:** 130–139 / 80–89 mmHg\n"
            "  - **Stage 2 Hypertension:** ≥ 140 / ≥ 90 mmHg\n\n"
            "• **Actionable Tips:** Reducing dietary sodium to under 1,500 mg/day, regular aerobic walking, managing chronic stress, and limiting alcohol can lower systolic BP by 5–10 mmHg."
        )

    # 5. Cholesterol Questions
    if any(k in msg for k in ['cholesterol', 'chol', 'ldl', 'hdl', 'triglycerides']):
        return (
            "🧪 **Cholesterol and Atherosclerosis:**\n\n"
            "Cholesterol is a lipid substance carried by lipoproteins in your blood.\n\n"
            "• **Total Cholesterol Ranges:**\n"
            "  - **Desirable:** < 200 mg/dL\n"
            "  - **Borderline High:** 200–239 mg/dL\n"
            "  - **High:** ≥ 240 mg/dL\n\n"
            "• **Why it matters:** Excess LDL ('bad cholesterol') penetrates damaged arterial walls, oxidizes, and forms fatty plaques (atherosclerosis) that restrict coronary blood flow.\n"
            "• **Dietary Interventions:** Eat oats, legumes, walnuts, and fatty fish (salmon), and limit saturated animal fats."
        )

    # 6. Chest Pain / Angina Questions
    if any(k in msg for k in ['chest pain', 'angina', 'cp', 'exang']):
        return (
            "🫀 **Understanding Angina Pectoris (Chest Pain Types):**\n\n"
            "Angina is discomfort caused by reduced blood flow to cardiac muscle.\n\n"
            "• **Four Clinical Classes Used in this Model:**\n"
            "  1. **Typical Angina:** Classic retrosternal chest pressure triggered by physical exertion or stress and relieved by rest or nitroglycerin.\n"
            "  2. **Atypical Angina:** Discomfort that exhibits only 2 of the 3 typical characteristics.\n"
            "  3. **Non-Anginal Pain:** Chest discomfort unlikely to be cardiac in origin (e.g. musculoskeletal or esophageal reflux).\n"
            "  4. **Asymptomatic:** No chest pain reported (frequent in diabetic patients with silent ischemia).\n\n"
            "• **Exercise-Induced Angina (`exang`):** If chest tightness arises specifically under physical exertion, it strongly correlates with critical coronary stenosis."
        )

    # 7. Thalassemia / Perfusion Defect Questions
    if any(k in msg for k in ['thal', 'thalassemia', 'reversible', 'fixed defect']):
        return (
            "🔬 **Thalassemia / Thallium Stress Perfusion:**\n\n"
            "In cardiac stress testing, nuclear thallium scintigraphy examines myocardial perfusion (blood flow distribution):\n\n"
            "• **Normal (1):** Uniform blood perfusion throughout the heart tissue.\n"
            "• **Fixed Defect (2):** Area that receives no blood flow both during exercise and at rest—typically indicates scar tissue from a prior myocardial infarction (heart attack).\n"
            "• **Reversible Defect (3):** Area that exhibits reduced blood flow during exercise but recovers at rest—indicates viable heart muscle that is actively ischemic and at high risk."
        )

    # 8. Diet & Nutrition Tips
    if any(k in msg for k in ['diet', 'food', 'eat', 'nutrition', 'salt', 'sugar']):
        return (
            "🥗 **Cardiovascular Nutrition Guide:**\n\n"
            "1. **The Mediterranean Standard:** Focus on extra virgin olive oil, wild fatty fish, colorful vegetables, berries, and raw nuts.\n"
            "2. **Sodium Reduction:** Keep daily sodium below 1,500–2,000 mg to prevent fluid retention and arterial stiffness.\n"
            "3. **Fiber Power:** Soluble fiber acts like a sponge in your digestive tract, binding to cholesterol particles and escorting them out.\n"
            "4. **Minimize Added Sugars:** Elevated fasting blood sugar causes endothelial inflammation and accelerates plaque calcification."
        )

    # 9. Model Architecture & Accuracy Questions
    if any(k in msg for k in ['how model works', 'algorithm', 'random forest', 'accuracy', 'dataset', 'machine learning']):
        return (
            "🤖 **CardioSense AI Architecture & Validation:**\n\n"
            "• **Algorithm:** Regularized Random Forest Classifier (150 decision trees, maximum depth 6, minimum leaf size 2).\n"
            "• **Dataset:** 1,190 multi-center patient records from the Statlog-Cleveland-Hungary combined cardiac database.\n"
            "• **Validation Rigor:** 5-Fold Stratified Cross-Validation.\n"
            "• **Performance Metrics:**\n"
            "  - **Cross-Validation Accuracy:** 87.65%\n"
            "  - **Holdout Test Accuracy:** 89.50%\n"
            "  - **Clinical Sensitivity / Recall:** 90.48%\n"
            "  - **ROC-AUC Score:** 95.53%"
        )

    # Default Helpful Response
    return (
        "💡 **CardioSense AI Assistant:**\n\n"
        "I can help explain your risk analysis and cardiovascular parameters. You can ask me:\n\n"
        "• *'What does high ST depression (oldpeak) mean?'*\n"
        "• *'How does my cholesterol and blood pressure affect my score?'*\n"
        "• *'What are the 4 types of chest pain in this screening?'*\n"
        "• *'What dietary and exercise changes improve heart health?'*\n"
        "• *'Explain the difference between fixed and reversible thal defects.'*\n"
        "• *'What is the accuracy and architecture of this ML model?'*\n\n"
        "What cardiovascular health question would you like me to answer?"
    )

