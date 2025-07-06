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
        
        # Fill missing values (from notebook logic) - fix pandas FutureWarning
        for col in ['Age', 'Income', 'Credit_Score']:
            if col in df.columns:
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value.iloc[0])
        
        for col in ['Loan_Amount']:
            if col in df.columns:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        # Define categorical columns for encoding
        categorical_cols = ['Gender', 'Race', 'Age_Group', 'Employment_Type', 
                           'Education_Level', 'Citizenship_Status', 'Language_Proficiency', 
                           'Disability_Status', 'Criminal_Record', 'Zip_Code_Group']
        
        # Create mapping dictionaries to preserve original value mappings
        encoding_mappings = {}
        
        # Encode categorical variables using LabelEncoder
        le = LabelEncoder()
        cols_to_encode = categorical_cols + (['Loan_Approved'] if is_train else [])
        
        for col in cols_to_encode:
            if col in df.columns:
                # Store original values before encoding
                original_values = df[col].copy()
                
                # Handle NaN values properly before encoding
                if original_values.isnull().any():
                    # Fill NaN with a placeholder that can be sorted
                    original_values = original_values.fillna('Missing')
                
                # Encode the column (LabelEncoder handles NaN automatically)
                df[col] = le.fit_transform(original_values)
                
                # Create mapping for bias analysis later - convert numpy int64 to Python int
                try:
                    unique_originals = sorted(original_values.unique())
                    unique_encoded = sorted(df[col].unique())
                    # Convert numpy int64 to Python int for JSON serialization
                    encoding_mappings[col] = {int(enc): orig for enc, orig in zip(unique_encoded, unique_originals)}
                except TypeError as e:
                    # Handle cases where values can't be sorted (mixed types)
                    logger.warning(f"Could not sort values for {col}: {e}")
                    unique_originals = list(original_values.unique())
                    unique_encoded = sorted(df[col].unique())
                    encoding_mappings[col] = {int(enc): orig for enc, orig in zip(unique_encoded, unique_originals)}
                
                # Log the mapping for debugging
                if col in ['Gender', 'Race', 'Age_Group', 'Citizenship_Status']:
                    logger.info(f"{col} encoding mapping:")
                    for enc, orig in encoding_mappings[col].items():
                        logger.info(f"  {enc} -> {orig}")
        
        return df, encoding_mappings
    
    # Process training data
    df_train_processed, train_mappings = preprocess_data(df_train, is_train=True)
    
    # Process test data  
    df_test_processed, test_mappings = preprocess_data(df_test, is_train=False)
    
    # Save processed training data
    df_train_processed.to_csv(output_path, index=False)
    logger.info(f"Processed training data saved to: {output_path}")
    
    # Save processed test data
    test_output_path = PROCESSED_DATA_DIR / "test_processed.csv"
    df_test_processed.to_csv(test_output_path, index=False)
    logger.info(f"Processed test data saved to: {test_output_path}")
    
    # Save encoding mappings for bias analysis
    import json
    mappings_path = PROCESSED_DATA_DIR / "encoding_mappings.json"
    with open(mappings_path, 'w') as f:
        json.dump(train_mappings, f, indent=2)
    logger.info(f"Encoding mappings saved to: {mappings_path}")
    
    logger.success("Dataset processing complete.")


if __name__ == "__main__":
    app()
