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
