import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score,
                             RocCurveDisplay, PrecisionRecallDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import xgboost as xgb

def prepare_data(features, target, test_size=0.2, random_state=42):
    """Split and scale data."""

    # --- Use data from vae as training and the remaining as test ---
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def objective_logistic(params, X_train, y_train, cv=5):
    """Objective function for Logistic Regression with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    model = LogisticRegression(
        C=params['C'],
        max_iter=params['max_iter'],
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    return {'loss': -scores.mean(), 'status': STATUS_OK}

def objective_rf(params, X_train, y_train, cv=5):
    """Objective function for Random Forest with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    return {'loss': -scores.mean(), 'status': STATUS_OK}

def objective_xgb(params, X_train, y_train, cv=5):
    """Objective function for XGBoost with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        random_state=42,
        eval_metric='logloss'
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    return {'loss': -scores.mean(), 'status': STATUS_OK}

def objective_nn(params, X_train, y_train, cv=5):
    """Objective function for Neural Network with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    model = MLPClassifier(
        hidden_layer_sizes=(int(params['hidden_layer_1']), int(params['hidden_layer_2'])),
        learning_rate_init=params['learning_rate'],
        alpha=params['alpha'],
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    return {'loss': -scores.mean(), 'status': STATUS_OK}

def tune_hyperparameters(model_name, X_train, y_train, max_evals=100, n_folds=5):
    """Tune hyperparameters using hyperopt with stratified cross-validation."""
    # cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    if model_name == 'logistic':
        space = {
            'C': hp.loguniform('C', -5, 3),
            'max_iter': hp.choice('max_iter', [100, 500, 1000])
        }
        obj = lambda params: objective_logistic(params, X_train, y_train, cv=n_folds)
    elif model_name == 'rf':
        space = {
            'n_estimators': hp.randint('n_estimators', 50, 300),
            'max_depth': hp.randint('max_depth', 5, 30),
            'min_samples_split': hp.randint('min_samples_split', 2, 20)
        }
        obj = lambda params: objective_rf(params, X_train, y_train, cv=n_folds)
    elif model_name == 'gb':
        space = {
            'n_estimators': hp.randint('n_estimators', 50, 300),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'max_depth': hp.randint('max_depth', 3, 15)
        }
        obj = lambda params: objective_xgb(params, X_train, y_train, cv=n_folds)
    elif model_name == 'nn':
        space = {
            'hidden_layer_1': hp.randint('hidden_layer_1', 32, 256),
            'hidden_layer_2': hp.randint('hidden_layer_2', 16, 128),
            'learning_rate': hp.loguniform('learning_rate', -4, -1),
            'alpha': hp.loguniform('alpha', -6, -2)
        }
        obj = lambda params: objective_nn(params, X_train, y_train, cv=n_folds)
    
    trials = Trials()
    best = fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best, trials

def build_model(name, params):
    if name == 'logistic':
        return LogisticRegression(**params)
    if name == 'rf':
        return RandomForestClassifier(**params)
    if name == 'xgb':
        return xgb.XGBClassifier(**params)
    if name == 'nn':
        return MLPClassifier(**params)
    raise ValueError(f"Unknown model name: {name}")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, cv=5):
    """Comprehensive model evaluation with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Stratified cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Test set predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    
    print(f"\n--- Stratified Cross-Validation Scores (5-Fold) ---")
    print(f"ROC-AUC CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print(f"\n--- Test Set Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    # print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return y_pred, y_pred_proba

def plot_results(y_test, y_pred_proba, model_name, ax1, ax2):
    """Plot ROC curve and confusion matrix."""
    # fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    # ax1.set_title(f'{model_name} - ROC Curve')
    ax1.legend()
    ax1.grid()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax2.plot(recall, precision, label=f'{model_name} (AP = {auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    # ax2.set_title(f'{model_name} - Precision-Recall Curve')
    ax2.grid()
    
    plt.tight_layout()
    # plt.show()
