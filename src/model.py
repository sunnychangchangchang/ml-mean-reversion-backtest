"""
Machine learning module.
Supports Logistic Regression (L1) and XGBoost for intraday direction prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import logging
from typing import List, Tuple, Dict, Optional, Union

logger = logging.getLogger(__name__)

Model = Union[LogisticRegression, XGBClassifier]


def train_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    train_cutoff_date: str,
    model_type: str = 'lr',
) -> Tuple[Model, StandardScaler, pd.DataFrame]:
    """
    Train a classification model using a chronological split.

    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        train_cutoff_date: Date string for train/test split (YYYY-MM-DD)
        model_type: 'lr' for Logistic Regression, 'xgb' for XGBoost

    Returns:
        Tuple of (trained model, fitted scaler, training data)
    """
    model_data = df.dropna(subset=feature_columns + ['target'])

    train_data = model_data[model_data.index.get_level_values('Date') < train_cutoff_date]
    test_data  = model_data[model_data.index.get_level_values('Date') >= train_cutoff_date]

    if len(train_data) == 0:
        raise ValueError("No training data available")
    if len(test_data) == 0:
        raise ValueError("No test data available")

    X_train = train_data[feature_columns].values
    y_train = train_data['target'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if model_type == 'lr':
        # L1 penalty automatically zeros out irrelevant features.
        # liblinear is required for L1; C=0.5 = moderate regularization.
        model = LogisticRegression(
            penalty='l1',
            C=0.5,
            solver='liblinear',
            random_state=42,
            max_iter=1000,
        )
        model.fit(X_train_scaled, y_train)

    elif model_type == 'xgb':
        # Stronger regularization for small financial datasets:
        # max_depth=2 and min_child_weight=5 prevent splits on noisy subsets;
        # gamma=1 requires a meaningful loss reduction before each split;
        # reg_alpha=0.1 adds L1 shrinkage on leaf weights.
        model = XGBClassifier(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1.0,
            reg_alpha=0.1,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train_scaled, y_train)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'lr' or 'xgb'.")

    logger.info(f"[{model_type.upper()}] trained on {len(X_train)} samples up to {train_cutoff_date}")
    return model, scaler, train_data


def predict_proba(
    model: Model,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Generate prediction probabilities. Works for both LR and XGBoost.

    Returns:
        DataFrame with ml_probability column added.
    """
    pred_data = df.dropna(subset=feature_columns)
    if len(pred_data) == 0:
        raise ValueError("No data available for prediction")

    X = pred_data[feature_columns].values
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]

    result = pred_data.copy()
    result['ml_probability'] = proba
    return result


def get_feature_importance(
    model: Model,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Return feature importance for LR (coefficients) or XGBoost (gain-based).
    Column names differ so the UI can label them appropriately.
    """
    if hasattr(model, 'coef_'):
        # Logistic Regression
        importance = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': model.coef_[0],
        })
        importance['abs_coefficient'] = importance['coefficient'].abs()
        return importance.sort_values('abs_coefficient', ascending=False)
    else:
        # XGBoost — feature_importances_ uses 'gain' by default in sklearn API
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_,
        })
        return importance.sort_values('importance', ascending=False)


def check_calibration_oos(
    proba: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> Optional[Dict]:
    """
    Check calibration using out-of-sample walk-forward predictions.
    proba / actual are collected across all test folds — fully OOS.
    """
    if len(proba) == 0:
        return None
    frac_pos, mean_pred = calibration_curve(actual, proba, n_bins=n_bins, strategy='quantile')
    brier = float(brier_score_loss(actual, proba))
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    return {
        'fraction_of_positives': frac_pos,
        'mean_predicted_value': mean_pred,
        'brier_score': brier,
        'ece': ece,
        'n_samples': len(actual),
        'base_rate': float(actual.mean()),
    }


def check_calibration(
    model: Model,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_columns: List[str],
    n_bins: int = 10,
) -> Optional[Dict]:
    """Check probability calibration on df (typically the training split)."""
    valid = df.dropna(subset=feature_columns + ['target'])
    if len(valid) == 0:
        return None

    X = scaler.transform(valid[feature_columns].values)
    y = valid['target'].values
    proba = model.predict_proba(X)[:, 1]

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy='quantile')
    brier = float(brier_score_loss(y, proba))
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

    return {
        'fraction_of_positives': frac_pos,
        'mean_predicted_value': mean_pred,
        'brier_score': brier,
        'ece': ece,
        'n_samples': len(valid),
        'base_rate': float(y.mean()),
    }
