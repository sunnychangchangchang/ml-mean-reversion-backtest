"""
Machine learning module.
Trains Logistic Regression to predict intraday direction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


def train_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    train_cutoff_date: str
) -> Tuple[LogisticRegression, StandardScaler, pd.DataFrame]:
    """
    Train Logistic Regression model using chronological split.
    
    IMPORTANT: No look-ahead bias - we only use data available up to day t
    to predict day t+1 outcomes.
    
    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        train_cutoff_date: Date string for train/test split (YYYY-MM-DD)
    
    Returns:
        Tuple of (trained model, fitted scaler, training data)
    """
    # Drop rows with NaN in features or target
    model_data = df.dropna(subset=feature_columns + ['target'])
    
    # Split by date (chronological, no shuffling)
    train_data = model_data[model_data.index.get_level_values('Date') < train_cutoff_date]
    test_data = model_data[model_data.index.get_level_values('Date') >= train_cutoff_date]
    
    if len(train_data) == 0:
        raise ValueError("No training data available")
    
    if len(test_data) == 0:
        raise ValueError("No test data available")
    
    # Prepare features and target
    X_train = train_data[feature_columns].values
    y_train = train_data['target'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Training date range: {train_data.index.get_level_values('Date').min()} to {train_data.index.get_level_values('Date').max()}")
    
    return model, scaler, train_data


def predict_proba(
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Generate prediction probabilities for all data points.
    
    Args:
        model: Trained Logistic Regression model
        scaler: Fitted StandardScaler
        df: DataFrame with features
        feature_columns: List of feature column names
    
    Returns:
        DataFrame with probability predictions
    """
    # Drop rows with NaN in features
    pred_data = df.dropna(subset=feature_columns)
    
    if len(pred_data) == 0:
        raise ValueError("No data available for prediction")
    
    # Prepare features
    X = pred_data[feature_columns].values
    X_scaled = scaler.transform(X)
    
    # Get probability of class 1 (Close[t+1] > Open[t+1])
    proba = model.predict_proba(X_scaled)[:, 1]
    
    # Add predictions to dataframe
    result = pred_data.copy()
    result['ml_probability'] = proba
    
    return result


def get_feature_importance(
    model: LogisticRegression,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Get feature importance (coefficients) from Logistic Regression.
    
    Args:
        model: Trained Logistic Regression model
        feature_columns: List of feature column names
    
    Returns:
        DataFrame with feature names and coefficients
    """
    importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_[0]
    })
    importance['abs_coefficient'] = importance['coefficient'].abs()
    importance = importance.sort_values('abs_coefficient', ascending=False)

    return importance


def check_calibration_oos(
    proba: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> Optional[Dict]:
    """
    Check calibration using out-of-sample walk-forward predictions.
    proba / actual are collected across all test folds — fully OOS.
    This is more rigorous than in-sample calibration on training data.
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
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_columns: List[str],
    n_bins: int = 10,
) -> Optional[Dict]:
    """
    Check probability calibration on df (typically the training split).

    Logistic Regression minimises log-loss (a proper scoring rule), so it is
    theoretically well-calibrated. This function provides empirical evidence
    to support — or challenge — using 0.5 as the decision threshold.

    Returns a dict with:
      fraction_of_positives  : actual positive rate per probability bin
      mean_predicted_value   : mean predicted probability per bin
      brier_score            : mean squared error of probabilities (lower = better)
      ece                    : Expected Calibration Error (mean |pred - actual|)
    """
    valid = df.dropna(subset=feature_columns + ['target'])
    if len(valid) == 0:
        return None

    X = scaler.transform(valid[feature_columns].values)
    y = valid['target'].values
    proba = model.predict_proba(X)[:, 1]

    frac_pos, mean_pred = calibration_curve(
        y, proba, n_bins=n_bins, strategy='quantile'
    )
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