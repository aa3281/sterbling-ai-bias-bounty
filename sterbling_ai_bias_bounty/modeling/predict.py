from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import joblib
import numpy as np

from sterbling_ai_bias_bounty.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "submission.csv",
    # -----------------------------------------
):
    """Generate predictions using trained model."""
    logger.info("Performing inference for loan approval model...")
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please train a model first:")
        logger.error("  python loan_model.py train-model")
        return
    
    # Load test features
    if not features_path.exists():
        # Try alternative path
        features_path = PROCESSED_DATA_DIR / "test_processed.csv"
        if not features_path.exists():
            logger.error(f"Test features file not found: {features_path}")
            logger.error("Please run feature engineering on test data first:")
            logger.error("  python loan_model.py feature-engineering")
            return
    
    X_test = pd.read_csv(features_path)
    logger.info(f"Test features shape: {X_test.shape}")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler_path = MODELS_DIR / "scaler.pkl"
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        # Check if model needs scaled features (Logistic Regression)
        if hasattr(model, 'coef_'):
            X_test_processed = scaler.transform(X_test.drop('ID', axis=1, errors='ignore'))
        else:
            X_test_processed = X_test.drop('ID', axis=1, errors='ignore')
    else:
        X_test_processed = X_test.drop('ID', axis=1, errors='ignore')
    
    # Generate predictions (from notebook logic)
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Create submission format (from notebook)
    submission = pd.DataFrame({
        'ID': X_test['ID'] if 'ID' in X_test.columns else range(len(X_test)),
        'Loan_Approved': ['Approved' if pred == 1 else 'Denied' for pred in y_pred]
    })
    
    # Save predictions (matching notebook's submission.csv format)
    submission.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # Additional analysis from notebook
    # Save detailed predictions with probabilities for analysis
    detailed_predictions = pd.DataFrame({
        'ID': X_test['ID'] if 'ID' in X_test.columns else range(len(X_test)),
        'Loan_Approved': ['Approved' if pred == 1 else 'Denied' for pred in y_pred],
        'Approval_Probability': y_pred_proba,
        'Prediction_Confidence': np.maximum(y_pred_proba, 1 - y_pred_proba)
    })
    
    detailed_path = PROCESSED_DATA_DIR / "detailed_predictions.csv"
    detailed_predictions.to_csv(detailed_path, index=False)
    logger.info(f"Detailed predictions saved to: {detailed_path}")

    # Log prediction statistics (from notebook)
    approval_rate = (y_pred == 1).mean()
    logger.info(f"Approval rate: {approval_rate:.2%}")
    logger.info(f"Average approval probability: {y_pred_proba.mean():.4f}")
    
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
