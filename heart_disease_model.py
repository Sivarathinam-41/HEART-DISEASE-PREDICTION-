import os
import json
import pickle
import pandas as pd
import numpy as np

# Feature definitions matching training schema
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Clinical valid ranges for safety checks
FEATURE_BOUNDS = {
    'age': (18, 120),
    'sex': (0, 1),
    'cp': (0, 3),
    'trestbps': (60, 250),
    'chol': (100, 600),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalach': (50, 250),
    'exang': (0, 1),
    'oldpeak': (0.0, 7.0),
    'slope': (0, 2),
    'ca': (0, 3),
    'thal': (1, 3)
}

_MODEL_CACHE = None
_METRICS_CACHE = None

def get_model():
    """Load and cache the trained Pipeline."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train_model.py first.")
        with open(model_path, 'rb') as f:
            _MODEL_CACHE = pickle.load(f)
    return _MODEL_CACHE

def get_metrics():
    """Load cached evaluation metrics."""
    global _METRICS_CACHE
    if _METRICS_CACHE is None:
        metrics_path = os.path.join(os.path.dirname(__file__), 'model_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                _METRICS_CACHE = json.load(f)
        else:
            _METRICS_CACHE = {}
    return _METRICS_CACHE

def validate_and_clean_input(raw_input):
    """
    Ensure input is structured as a dictionary with all 13 features
    and numeric types within expected physiological ranges.
    """
    if isinstance(raw_input, (list, tuple)):
        if len(raw_input) != len(FEATURE_ORDER):
            raise ValueError(f"Expected {len(FEATURE_ORDER)} features, received {len(raw_input)}.")
        raw_dict = {feat: raw_input[i] for i, feat in enumerate(FEATURE_ORDER)}
    elif isinstance(raw_input, dict):
        raw_dict = raw_input
    else:
        raise TypeError("Input must be a dictionary of feature names or a list of values.")

    cleaned = {}
    for feat in FEATURE_ORDER:
        if feat not in raw_dict:
            raise ValueError(f"Missing required clinical parameter: '{feat}'")
        try:
            val = float(raw_dict[feat])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value for '{feat}': {raw_dict[feat]}")

        # Check reasonable bounds
        min_v, max_v = FEATURE_BOUNDS[feat]
        if val < min_v or val > max_v:
            # Soft clamp or raise warning
            val = max(min_v, min(max_v, val))
        cleaned[feat] = val

    return cleaned

def explain_risk_factors(data):
    """
    Generate patient-specific clinical factor insights based on input values.
    """
    factors = []
    if data['trestbps'] >= 140:
        factors.append(f"Elevated resting blood pressure ({int(data['trestbps'])} mmHg)")
    if data['chol'] >= 240:
        factors.append(f"High serum cholesterol ({int(data['chol'])} mg/dl)")
    if data['cp'] in [1, 2, 3]:
        cp_labels = {1: "Atypical Angina", 2: "Non-anginal Chest Pain", 3: "Asymptomatic"}
        factors.append(f"Reported chest pain indicator ({cp_labels.get(int(data['cp']))})")
    if data['exang'] == 1:
        factors.append("Exercise-induced angina present")
    if data['oldpeak'] >= 1.5:
        factors.append(f"Significant ST depression ({data['oldpeak']} mm)")
    if data['ca'] >= 1:
        factors.append(f"Fluoroscopy shows {int(data['ca'])} major vessel(s) affected")
    if data['thal'] == 3:
        factors.append("Reversible defect identified in thalassemia screening")
    if data['thalach'] < 120 and data['age'] < 60:
        factors.append(f"Lower maximum heart rate response ({int(data['thalach'])} bpm)")

    if not factors:
        factors.append("All key baseline cardiovascular parameters within standard observational limits")
    return factors

def predict_heart_disease(raw_input):
    """
    Primary prediction function.
    Returns structured prediction, probabilities, risk tiers, and explanatory insights.
    """
    cleaned = validate_and_clean_input(raw_input)
    pipeline = get_model()

    # Pass DataFrame with feature names to ensure ColumnTransformer alignment
    df_input = pd.DataFrame([cleaned], columns=FEATURE_ORDER)

    prediction = int(pipeline.predict(df_input)[0])
    prob = float(pipeline.predict_proba(df_input)[0][1])
    risk_percentage = round(prob * 100, 1)

    # 3-tier clinical classification
    if risk_percentage < 35.0:
        risk_level = "Low Risk"
        badge_class = "success"
        color = "text-success"
        icon = "❤️"
        summary = "Statistical probability indicates low likelihood of coronary artery disease."
    elif risk_percentage < 65.0:
        risk_level = "Moderate Risk"
        badge_class = "warning"
        color = "text-warning"
        icon = "⚠️"
        summary = "Intermediate cardiovascular risk indicators detected. Follow-up diagnostic screening recommended."
    else:
        risk_level = "High Risk"
        badge_class = "danger"
        color = "text-danger"
        icon = "🚨"
        summary = "Elevated indicators correlated with significant coronary narrowing (>50%). Consult a cardiologist."

    risk_factors = explain_risk_factors(cleaned)

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
        'input_data': cleaned
    }

def get_demo_presets():
    """Standard pre-filled patient profiles for demonstration and testing."""
    return {
        'healthy': {
            'label': 'Low-Risk Healthy Profile',
            'data': {
                'age': 38, 'sex': 0, 'cp': 0, 'trestbps': 118,
                'chol': 180, 'fbs': 0, 'restecg': 0, 'thalach': 175,
                'exang': 0, 'oldpeak': 0.2, 'slope': 2, 'ca': 0, 'thal': 2
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

if __name__ == '__main__':
    # Self-test
    presets = get_demo_presets()
    print("Testing Healthy Preset:")
    res_h = predict_heart_disease(presets['healthy']['data'])
    print(f"  Result: {res_h['risk_level']} ({res_h['risk_percentage']}%)")

    print("\nTesting High Risk Preset:")
    res_hr = predict_heart_disease(presets['high_risk']['data'])
    print(f"  Result: {res_hr['risk_level']} ({res_hr['risk_percentage']}%)")
