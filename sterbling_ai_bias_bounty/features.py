from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from sterbling_ai_bias_bounty.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    """Generate features from processed dataset."""
    logger.info("Generating features from dataset...")
    
    # Load processed data
    df = pd.read_csv(input_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Feature engineering from notebook Cell 5 (minimal - mostly commented out)
    # The notebook does NOT do extensive feature engineering
    
    # Log transformation for skewed numerical features (commented out in notebook)
    # if 'Loan_Amount' in df.columns:
    #     df['Loan_Amount_Log'] = np.log1p(df['Loan_Amount'])
    
    # if 'Income' in df.columns:
    #     df['Income_Log'] = np.log1p(df['Income'])
    
    # Create interaction features (not in notebook)
    # if 'Income' in df.columns and 'Credit_Score' in df.columns:
    #     df['Income_Credit_Ratio'] = df['Income'] / (df['Credit_Score'] + 1)
    
    # if 'Loan_Amount' in df.columns and 'Income' in df.columns:
    #     df['Loan_Income_Ratio'] = df['Loan_Amount'] / (df['Income'] + 1)
    
    # Age-based features (not in notebook)
    # if 'Age' in df.columns:
    #     df['Age_Squared'] = df['Age'] ** 2
    #     df['Age_Binned'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
    #                              labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
    #     df['Age_Binned'] = df['Age_Binned'].cat.codes
    
    # Credit score based features (not in notebook)
    # if 'Credit_Score' in df.columns:
    #     df['Credit_Score_Squared'] = df['Credit_Score'] ** 2
    #     df['Credit_Score_Binned'] = pd.cut(df['Credit_Score'], bins=[0, 300, 600, 700, 850], 
    #                                       labels=['Poor', 'Fair', 'Good', 'Excellent'])
    #     df['Credit_Score_Binned'] = df['Credit_Score_Binned'].cat.codes
    
    # Income based features (not in notebook)
    # if 'Income' in df.columns:
    #     df['Income_Squared'] = df['Income'] ** 2
    #     df['High_Income'] = (df['Income'] > df['Income'].quantile(0.75)).astype(int)
    
    # Loan amount based features (not in notebook)
    # if 'Loan_Amount' in df.columns:
    #     df['Large_Loan'] = (df['Loan_Amount'] > df['Loan_Amount'].quantile(0.75)).astype(int)
    
    # Interaction features for bias detection (not in notebook)
    # if 'Gender' in df.columns and 'Income' in df.columns:
    #     df['Gender_Income_Interaction'] = df['Gender'] * df['Income']
    
    # if 'Race' in df.columns and 'Credit_Score' in df.columns:
    #     df['Race_Credit_Interaction'] = df['Race'] * df['Credit_Score']

    # Keep original features structure like notebook - NO additional engineering
    
    # Separate features and target exactly like notebook Cell 6
    if 'Loan_Approved' in df.columns:
        # Training data - exclude ID and target from features (14 features like notebook)
        X = df.drop(['Loan_Approved', 'ID'], axis=1, errors='ignore')
        y = df['Loan_Approved']
        
        # Save features and labels separately
        X.to_csv(output_path, index=False)
        y.to_csv(PROCESSED_DATA_DIR / "labels.csv", index=False)
        logger.info(f"Features saved to: {output_path}")
        logger.info(f"Labels saved to: {PROCESSED_DATA_DIR / 'labels.csv'}")
    else:
        # Test data - exclude ID from features but keep same structure
        feature_cols = [col for col in df.columns if col != 'ID']
        X = df[feature_cols]
        X.to_csv(output_path, index=False)
        logger.info(f"Test features saved to: {output_path}")
    
    logger.success("Feature generation complete.")


if __name__ == "__main__":
    app()
    app()
