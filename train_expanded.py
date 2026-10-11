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

NUMERIC_FEATURES = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 'exercise_angina', 'st_slope']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = 'target'

def load_data(csv_path='heart_expanded.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} not found.")

    df = pd.read_csv(csv_path)
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # In combined datasets, cholesterol = 0 indicates missing recorded value
    chol_median = df[df['cholesterol'] > 0]['cholesterol'].median()
    df['cholesterol'] = df['cholesterol'].replace(0, chol_median)

    # resting_bp_s = 0 indicates missing recorded value
    bp_median = df[df['resting_bp_s'] > 0]['resting_bp_s'].median()
    df['resting_bp_s'] = df['resting_bp_s'].replace(0, bp_median)

    return df

def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', 'passthrough', CATEGORICAL_FEATURES)
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

def train():
    print("[*] Loading Expanded Heart Disease Dataset (1,190 records)...", flush=True)
    df = load_data('heart_expanded.csv')
    print(f"[OK] Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns", flush=True)

    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    print("[*] Running 5-Fold Stratified Cross-Validation...", flush=True)
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

    print(f"  - Train Accuracy: {cv_metrics['train_accuracy_mean']*100:.2f}%", flush=True)
    print(f"  - CV Accuracy:    {cv_metrics['cv_accuracy_mean']*100:.2f}% (+/- {cv_metrics['cv_accuracy_std']*100:.2f}%)", flush=True)
    print(f"  - CV Sensitivity: {cv_metrics['cv_recall_mean']*100:.2f}%", flush=True)
    print(f"  - CV Precision:   {cv_metrics['cv_precision_mean']*100:.2f}%", flush=True)
    print(f"  - CV F1-Score:    {cv_metrics['cv_f1_mean']*100:.2f}%", flush=True)
    print(f"  - CV ROC-AUC:     {cv_metrics['cv_roc_auc_mean']*100:.2f}%", flush=True)

    print("\n[*] Evaluating Holdout Test Split (80/20 Stratified)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    holdout_pipeline = build_pipeline()
    holdout_pipeline.fit(X_train, y_train)

    y_pred = holdout_pipeline.predict(X_test)
    y_prob = holdout_pipeline.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_prob)

    print(f"  - Holdout Test Accuracy: {test_acc*100:.2f}%", flush=True)
    print(f"  - Holdout Test Recall:   {test_rec*100:.2f}%", flush=True)
    print(f"  - Holdout Test ROC-AUC:  {test_roc*100:.2f}%", flush=True)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred), flush=True)

    print("\n[*] Training production pipeline on full 1,190 dataset...", flush=True)
    prod_pipeline = build_pipeline()
    prod_pipeline.fit(X, y)

    # Save artifacts
    with open('heart_disease_expanded_model.pkl', 'wb') as f:
        pickle.dump(prod_pipeline, f)
    print("[OK] Saved 'heart_disease_expanded_model.pkl'", flush=True)

    metrics_payload = {
        'dataset_name': 'Statlog-Cleveland-Hungary Expanded (1,190 records)',
        'total_samples': int(len(df)),
        'cross_validation': cv_metrics,
        'holdout_test': {
            'accuracy': round(float(test_acc), 4),
            'recall': round(float(test_rec), 4),
            'roc_auc': round(float(test_roc), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        },
        'feature_order': ALL_FEATURES
    }

    with open('model_metrics_expanded.json', 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print("[OK] Saved 'model_metrics_expanded.json'", flush=True)

if __name__ == '__main__':
    train()
