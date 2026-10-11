import os
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Feature definitions
NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = 'target'

def load_and_clean_data(csv_path='heart.csv'):
    """Load dataset and handle known anomalies in UCI Cleveland data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Impute UCI Cleveland artifact codes:
    # 'ca' has four values of 4 (missing code in original data) -> impute with mode (0)
    # 'thal' has two values of 0 (missing code in original data) -> impute with mode (2)
    df['ca'] = df['ca'].replace(4, df['ca'].mode()[0])
    df['thal'] = df['thal'].replace(0, df['thal'].mode()[0])

    return df

def build_pipeline():
    """Create a unified scikit-learn pipeline with preprocessor and classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', 'passthrough', CATEGORICAL_FEATURES)
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def train_and_evaluate():
    print("[*] Loading and preparing Heart Disease dataset...")
    df = load_and_clean_data('heart.csv')
    print(f"[OK] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # NOTE ON TARGET ENCODING:
    # In the Kaggle Cleveland dataset, target=1 corresponds to absence of disease (num=0 in UCI),
    # and target=0 corresponds to presence of disease (ca>0, oldpeak>2, exang=1).
    # To follow standard medical machine learning conventions, we align the target so that:
    # 1 = Heart Disease Detected (Positive Case / At Risk)
    # 0 = No Heart Disease / Healthy (Negative Case)
    X = df[ALL_FEATURES]
    y = 1 - df[TARGET_COL]

    # 1. Stratified 5-Fold Cross-Validation
    print("\n[*] Running 5-Fold Stratified Cross-Validation...")
    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=True
    )

    cv_metrics = {
        'train_accuracy_mean': round(float(np.mean(cv_results['train_accuracy'])), 4),
        'cv_accuracy_mean': round(float(np.mean(cv_results['test_accuracy'])), 4),
        'cv_accuracy_std': round(float(np.std(cv_results['test_accuracy'])), 4),
        'cv_precision_mean': round(float(np.mean(cv_results['test_precision'])), 4),
        'cv_recall_mean': round(float(np.mean(cv_results['test_recall'])), 4),
        'cv_f1_mean': round(float(np.mean(cv_results['test_f1'])), 4),
        'cv_roc_auc_mean': round(float(np.mean(cv_results['test_roc_auc'])), 4),
    }

    print(f"  - Train Accuracy (Mean): {cv_metrics['train_accuracy_mean'] * 100:.2f}%")
    print(f"  - CV Accuracy:           {cv_metrics['cv_accuracy_mean'] * 100:.2f}% (+/- {cv_metrics['cv_accuracy_std'] * 100:.2f}%)")
    print(f"  - CV Recall (Sensitivity): {cv_metrics['cv_recall_mean'] * 100:.2f}%")
    print(f"  - CV Precision:          {cv_metrics['cv_precision_mean'] * 100:.2f}%")
    print(f"  - CV F1-Score:           {cv_metrics['cv_f1_mean'] * 100:.2f}%")
    print(f"  - CV ROC-AUC:            {cv_metrics['cv_roc_auc_mean'] * 100:.2f}%")

    # 2. Train-Test Split (80/20 with Stratification) for Confusion Matrix
    print("\n[*] Evaluating Holdout Test Split (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    holdout_pipeline = build_pipeline()
    holdout_pipeline.fit(X_train, y_train)

    y_pred = holdout_pipeline.predict(X_test)
    y_prob = holdout_pipeline.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  - Holdout Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"  - Holdout Test Recall:    {test_rec * 100:.2f}%")
    print(f"  - Holdout Test ROC-AUC:   {test_roc_auc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 3. Fit Pipeline on full dataset for production deployment
    print("\n[*] Training production model on full dataset...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    # Extract feature importances
    rf_model = final_pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    feature_importance_map = sorted(
        [{'feature': feat, 'importance': round(float(imp), 4)} for feat, imp in zip(ALL_FEATURES, importances)],
        key=lambda x: x['importance'],
        reverse=True
    )

    # Save artifacts
    with open('heart_disease_model.pkl', 'wb') as f:
        pickle.dump(final_pipeline, f)
    print("[OK] Saved production pipeline to 'heart_disease_model.pkl'")

    # Save scaler separately for backward compatibility
    scaler = StandardScaler()
    scaler.fit(df[ALL_FEATURES])
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[OK] Saved fallback scaler to 'scaler.pkl'")

    # Save metrics JSON for web UI display
    metrics_payload = {
        'cross_validation': cv_metrics,
        'holdout_test': {
            'accuracy': round(float(test_acc), 4),
            'precision': round(float(test_prec), 4),
            'recall': round(float(test_rec), 4),
            'f1_score': round(float(test_f1), 4),
            'roc_auc': round(float(test_roc_auc), 4),
            'confusion_matrix': cm
        },
        'feature_importances': feature_importance_map,
        'feature_order': ALL_FEATURES,
        'dataset_summary': {
            'total_samples': int(len(df)),
            'positive_cases': int((y == 1).sum()),
            'negative_cases': int((y == 0).sum())
        }
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print("[OK] Saved metrics summary to 'model_metrics.json'")
    print("[READY] Pipeline training complete and ready for inference!")

if __name__ == '__main__':
    train_and_evaluate()
