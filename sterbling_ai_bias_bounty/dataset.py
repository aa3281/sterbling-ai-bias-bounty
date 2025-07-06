from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sterbling_ai_bias_bounty.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "loan_access_dataset.csv",
    test_path: Path = RAW_DATA_DIR / "test.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """Process raw loan dataset and prepare for modeling."""
    logger.info("Loading and processing loan dataset...")
    
    # Load training data
    df_train = pd.read_csv(train_path)
    logger.info(f"Training data shape: {df_train.shape}")
    
    # Load test data
    df_test = pd.read_csv(test_path)
    logger.info(f"Test data shape: {df_test.shape}")
    
    # Data preprocessing function from notebook
    def preprocess_data(df, is_train=True):
        df = df.copy()
        
        # Fill missing values (from notebook logic)
        for col in ['Age', 'Income', 'Credit_Score']:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        for col in ['Loan_Amount']:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Encode categorical variables (from notebook)
        categorical_cols = ['Gender', 'Race', 'Age_Group', 'Employment_Type', 
                          'Education_Level', 'Citizenship_Status', 'Language_Proficiency', 
                          'Disability_Status', 'Criminal_Record', 'Zip_Code_Group']
        
        if is_train:
            categorical_cols.append('Loan_Approved')
        
        le = LabelEncoder()
        for col in categorical_cols:
            if col in df.columns:
                df[col] = le.fit_transform(df[col])
        
        return df
    
    # Process training data
    df_train_processed = preprocess_data(df_train, is_train=True)
    
    # Process test data  
    df_test_processed = preprocess_data(df_test, is_train=False)
    
    # Save processed training data
    df_train_processed.to_csv(output_path, index=False)
    logger.info(f"Processed training data saved to: {output_path}")
    
    # Save processed test data
    test_output_path = PROCESSED_DATA_DIR / "test_processed.csv"
    df_test_processed.to_csv(test_output_path, index=False)
    logger.info(f"Processed test data saved to: {test_output_path}")
    
    logger.success("Dataset processing complete.")


if __name__ == "__main__":
    app()
