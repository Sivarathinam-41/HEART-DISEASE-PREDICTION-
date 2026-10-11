# 🫀 CardioSense AI — Heart Disease Prediction Web System

An end-to-end, clinical-grade Machine Learning web application designed to predict the likelihood of coronary heart disease based on 13 physiological and diagnostic markers from the UCI Cleveland dataset.

Built with **Python**, **Scikit-Learn**, **Flask**, **Bootstrap 5**, and modern **CSS Glassmorphism**, configured for deployment on **Vercel** and **Render**.

---

## 🌟 Key Features & Improvements

- **Unified Scikit-Learn Pipeline:** Uses `ColumnTransformer` to properly scale continuous numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) while preserving categorical/binary indicators, packaged into a single deployable artifact (`heart_disease_model.pkl`).
- **5-Fold Stratified Cross-Validation:** Validated against overfitting on the benchmark dataset, ensuring reproducible and statistically sound performance metrics.
- **Critical Target Inversion Correction:** Resolves a major widespread flaw in the Kaggle UCI Cleveland dataset where target encoding was inverted—preventing dangerous clinical false classifications.
- **Interactive Web Interface:** Beautiful glassmorphic UI featuring intuitive dropdowns, physiological validation constraints, one-click demo presets, dynamic probability meters, and patient-specific risk factor explanations.
- **Dual Interface (Web + REST API):** Supports both browser-based HTML forms and programmatic JSON endpoints (`/api/predict`, `/api/metrics`, `/api/presets`) ready for mobile or modern frontend integration.
- **Deployment Ready:** Pre-configured with `vercel.json`, `api/index.py` for Vercel Serverless Functions, and `Procfile` for Render / Railway.

---

## 📊 Model Architecture & Evaluation Metrics

| Metric | Score | Validation Methodology |
| :--- | :--- | :--- |
| **Cross-Validation Accuracy** | **83.49%** (± 5.21%) | 5-Fold Stratified Cross-Validation |
| **Sensitivity / Recall (Heart Disease)** | **90.91%** | 5-Fold Stratified Cross-Validation |
| **Precision** | **81.64%** | 5-Fold Stratified Cross-Validation |
| **F1-Score** | **85.85%** | 5-Fold Stratified Cross-Validation |
| **ROC-AUC Score** | **91.55%** | 5-Fold Stratified Cross-Validation |
| **Holdout Test Accuracy** | **83.61%** | 80/20 Stratified Split |
| **Holdout Test ROC-AUC** | **90.80%** | 80/20 Stratified Split |

- **Classifier:** Regularized Random Forest (`n_estimators=150`, `max_depth=5`, `min_samples_leaf=2`).
- **Training Samples:** 303 clinical records (UCI Cleveland Heart Disease benchmark).

---

## 📋 Clinical Feature Reference

| Feature | Medical Name | Data Type | Permissible Range / Values |
| :--- | :--- | :--- | :--- |
| `age` | Patient Age | Continuous | 20 – 100 years |
| `sex` | Biological Sex | Categorical | `1` = Male, `0` = Female |
| `cp` | Chest Pain Type | Categorical | `0` = Typical Angina, `1` = Atypical, `2` = Non-anginal, `3` = Asymptomatic |
| `trestbps` | Resting Blood Pressure | Continuous | 80 – 230 mmHg |
| `chol` | Serum Cholesterol | Continuous | 100 – 600 mg/dl |
| `fbs` | Fasting Blood Sugar | Binary | `0` = Normal (≤120 mg/dl), `1` = Elevated (>120 mg/dl) |
| `restecg` | Resting Electrocardiogram | Categorical | `0` = Normal, `1` = ST-T Abnormality, `2` = Left Ventricular Hypertrophy |
| `thalach` | Maximum Heart Rate | Continuous | 60 – 230 bpm |
| `exang` | Exercise Induced Angina | Binary | `0` = No, `1` = Yes |
| `oldpeak` | ST Depression | Continuous | 0.0 – 7.0 mm (relative to rest) |
| `slope` | ST Segment Slope | Categorical | `0` = Upsloping, `1` = Flat, `2` = Downsloping |
| `ca` | Fluoroscopy Vessels | Discrete | `0` to `3` major vessels colored |
| `thal` | Thalassemia Screening | Categorical | `1` = Normal, `2` = Fixed Defect, `3` = Reversible Defect |

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Pipeline
```bash
python train_model.py
```
This will run the 5-fold cross-validation, print the complete metrics report, and regenerate `heart_disease_model.pkl` and `model_metrics.json`.

### 4. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🌐 Deployment Instructions

### Deploying to Vercel
1. Install the Vercel CLI (or connect your GitHub repository to Vercel dashboard):
   ```bash
   npm install -g vercel
   vercel
   ```
2. The project contains `vercel.json` and `api/index.py`, which instruct Vercel to route traffic to the Flask application automatically.

### Deploying to Render
1. Push your project to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Connect your repository and configure:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`

---

## 🔌 REST API Documentation

### Predict Endpoint (`POST /api/predict`)

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65, "sex": 1, "cp": 3, "trestbps": 160,
    "chol": 286, "fbs": 1, "restecg": 2, "thalach": 108,
    "exang": 1, "oldpeak": 3.6, "slope": 1, "ca": 2, "thal": 3
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": 1,
    "probability": 0.804,
    "risk_percentage": 80.4,
    "risk_level": "High Risk",
    "badge_class": "danger",
    "icon": "🚨",
    "summary": "Elevated indicators correlated with significant coronary narrowing (>50%). Consult a cardiologist.",
    "risk_factors": [
      "Elevated resting blood pressure (160 mmHg)",
      "High serum cholesterol (286 mg/dl)",
      "Exercise-induced angina present",
      "Significant ST depression (3.6 mm)",
      "Fluoroscopy shows 2 major vessel(s) affected",
      "Reversible defect identified in thalassemia screening"
    ]
  }
}
```

---

## ⚖️ Educational Disclaimer
*This project is an academic simulation created for university evaluation. It is not an FDA-approved clinical diagnostic instrument. All medical predictions must be evaluated by licensed medical specialists.*
