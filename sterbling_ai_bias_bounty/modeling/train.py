from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import typer

from sterbling_ai_bias_bounty.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """Train multiple models and save the best performing one."""
    logger.info("Training loan approval models...")
    
    # Create directories if they don't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if input files exist
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return
    
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        return
    
    # Load features and labels
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).iloc[:, 0]  # First column
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to: {scaler_path}")
    
    models = {}
    
    # 1. Logistic Regression (from notebook)
    try:
        logger.info("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val_scaled)[:, 1])
        models['LogisticRegression'] = {'model': lr, 'auc': lr_auc}
        logger.info(f"Logistic Regression AUC: {lr_auc:.4f}")
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {e}")
    
    # 2. Random Forest with hyperparameter tuning (from notebook)
    try:
        logger.info("Training Random Forest with hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=42, n_jobs=1)
        param_grid_rf = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='roc_auc', n_jobs=1)
        grid_search_rf.fit(X_train, y_train)
        best_rf = grid_search_rf.best_estimator_
        
        rf_auc = roc_auc_score(y_val, best_rf.predict_proba(X_val)[:, 1])
        models['RandomForest'] = {'model': best_rf, 'auc': rf_auc}
        logger.info(f"Random Forest AUC: {rf_auc:.4f}")
        logger.info(f"Best RF params: {grid_search_rf.best_params_}")
        
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
    
    # 3. XGBoost (from notebook)
    try:
        from xgboost import XGBClassifier
        logger.info("Training XGBoost...")
        
        xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=1)
        param_grid_xgb = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1]
        }
        
        grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='roc_auc', n_jobs=1)
        grid_search_xgb.fit(X_train, y_train)
        best_xgb = grid_search_xgb.best_estimator_
        
        xgb_auc = roc_auc_score(y_val, best_xgb.predict_proba(X_val)[:, 1])
        models['XGBoost'] = {'model': best_xgb, 'auc': xgb_auc}
        logger.info(f"XGBoost AUC: {xgb_auc:.4f}")
        logger.info(f"Best XGB params: {grid_search_xgb.best_params_}")
        
    except ImportError:
        logger.warning("XGBoost not available. Install with: pip install xgboost")
        logger.warning("Or install all dependencies: pip install -e .")
    except Exception as e:
        logger.warning(f"XGBoost training failed: {e}, skipping...")
    
    # 4. LightGBM (from notebook)
    try:
        from lightgbm import LGBMClassifier
        logger.info("Training LightGBM...")
        
        lgbm = LGBMClassifier(random_state=42)
        param_grid_lgbm = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1]
        }
        
        grid_search_lgbm = GridSearchCV(lgbm, param_grid_lgbm, cv=3, scoring='roc_auc', n_jobs=1)
        grid_search_lgbm.fit(X_train, y_train)
        best_lgbm = grid_search_lgbm.best_estimator_
        
        lgbm_auc = roc_auc_score(y_val, best_lgbm.predict_proba(X_val)[:, 1])
        models['LightGBM'] = {'model': best_lgbm, 'auc': lgbm_auc}
        logger.info(f"LightGBM AUC: {lgbm_auc:.4f}")
        logger.info(f"Best LGBM params: {grid_search_lgbm.best_params_}")
        
    except ImportError:
        logger.warning("LightGBM not available. Install with: pip install lightgbm")
        logger.warning("Or install all dependencies: pip install -e .")
    except Exception as e:
        logger.warning(f"LightGBM training failed: {e}, skipping...")
    
    # Check if any models were trained
    if not models:
        logger.error("No models were successfully trained!")
        return
    
    # Select best model (from notebook logic)
    best_model_name = max(models.keys(), key=lambda k: models[k]['auc'])
    best_model = models[best_model_name]['model']
    best_auc = models[best_model_name]['auc']
    
    logger.info(f"Best model: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Save best model
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'best_model': best_model_name,
        'best_auc': best_auc,
        'all_models': {k: v['auc'] for k, v in models.items()}
    }
    
    import json
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.success("Model training complete.")


if __name__ == "__main__":
    app()
