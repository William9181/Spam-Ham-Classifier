import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


def build_svm_pipeline(max_features=20000, cv=3):
    """Build a text->SVM pipeline.

    If cv is a positive integer, the LinearSVC will be wrapped with
    CalibratedClassifierCV(cv=cv) to provide probability outputs. If cv is
    None or <= 0, a plain LinearSVC is used (no probability support) which is
    faster for quick experiments.
    """
    # Choose classifier: calibrated (provides predict_proba) or plain LinearSVC
    if cv is not None and int(cv) > 0:
        # Pass estimator positionally to be compatible with different sklearn versions
        clf = CalibratedClassifierCV(LinearSVC(), cv=int(cv))
    else:
        clf = LinearSVC()

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=max_features)),
        ('clf', clf)
    ])
    return pipe


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        probs = None

    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'roc_auc': None
    }
    if probs is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, probs))
        except Exception:
            metrics['roc_auc'] = None
    return metrics


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
