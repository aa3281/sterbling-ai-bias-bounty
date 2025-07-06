from pathlib import Path

# Handle missing dependencies gracefully
try:
    from loguru import logger
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install dependencies with:")
    print("  pip install -e .")
    exit(1)

try:
    import typer
except ImportError:
    print("Error: typer package not found. Please install dependencies with:")
    print("  pip install -e .")
    exit(1)

from sterbling_ai_bias_bounty.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, FIGURES_DIR
from sterbling_ai_bias_bounty import dataset, features, plots
from sterbling_ai_bias_bounty.modeling import train, predict

app = typer.Typer(help="AI Bias Bounty Loan Model Pipeline - Run individual steps or full pipeline")


@app.command()
def data(
    train_path: Path = RAW_DATA_DIR / "loan_access_dataset.csv",
    test_path: Path = RAW_DATA_DIR / "test.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """Process raw data into clean datasets."""
    logger.info("Running data processing step...")
    dataset.main(train_path, test_path, output_path)


@app.command()
def feature_engineering(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    """Generate features from processed data."""
    logger.info("Running feature engineering step...")
    features.main(input_path, output_path)


@app.command()
def train_model(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """Train the model."""
    logger.info("Running model training step...")
    train.main(features_path, labels_path, model_path)


@app.command()
def predict_model(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """Generate predictions using trained model."""
    logger.info("Running model prediction step...")
    predict.main(features_path, model_path, predictions_path)


@app.command()
def visualize(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = typer.Option(FIGURES_DIR / "analysis_plots.png", help="Output path for plot"),
):
    """Generate visualizations."""
    logger.info("Running visualization step...")
    plots.main(input_path, output_path)


def full_pipeline():
    """Run the complete pipeline from raw data to predictions and plots."""
    logger.info("🚀 Starting full pipeline execution...")
    
    # Check if raw data files exist
    train_path = RAW_DATA_DIR / "loan_access_dataset.csv"
    test_path = RAW_DATA_DIR / "test.csv"
    
    if not train_path.exists():
        logger.error(f"❌ Training data file not found: {train_path}")
        logger.error("Please place your data files in data/raw/ directory:")
        logger.error("  - data/raw/loan_access_dataset.csv")
        logger.error("  - data/raw/test.csv")
        logger.info("💡 You can download these files from the hackathon dataset")
        return
    
    if not test_path.exists():
        logger.error(f"❌ Test data file not found: {test_path}")
        logger.error("Please place your test.csv file in data/raw/ directory")
        return
    
    logger.info(f"✅ Found training data: {train_path}")
    logger.info(f"✅ Found test data: {test_path}")
    
    try:
        # Step 1: Data processing
        logger.info("📊 Step 1: Processing raw data...")
        data()
        logger.success("✅ Step 1 completed: Data processed")
        
        # Step 2: Feature engineering for training data
        logger.info("🔧 Step 2: Engineering features for training data...")
        feature_engineering()
        logger.success("✅ Step 2a completed: Training features engineered")
        
        # Step 2b: Feature engineering for test data
        logger.info("🔧 Step 2b: Engineering features for test data...")
        test_input_path = PROCESSED_DATA_DIR / "test_processed.csv"
        test_output_path = PROCESSED_DATA_DIR / "test_features.csv"
        if test_input_path.exists():
            features.main(test_input_path, test_output_path)
            logger.success("✅ Step 2b completed: Test features engineered")
        else:
            logger.warning(f"⚠️ Test processed data not found: {test_input_path}")
        
        # Step 3: Model training
        logger.info("🤖 Step 3: Training models...")
        train_model()
        logger.success("✅ Step 3 completed: Models trained")
        
        # Step 4: Generate predictions
        logger.info("🎯 Step 4: Generating predictions...")
        predict_model()
        logger.success("✅ Step 4 completed: Predictions generated")
        
        # Step 5: Create visualizations
        logger.info("📈 Step 5: Creating visualizations...")
        visualize()
        logger.success("✅ Step 5 completed: Visualizations created")
        
        logger.success("🎉 Full pipeline execution complete!")
        logger.info("Check the following directories for outputs:")
        logger.info(f"  📁 Processed data: {PROCESSED_DATA_DIR}")
        logger.info(f"  🤖 Models: {MODELS_DIR}")
        logger.info(f"  📊 Figures: {FIGURES_DIR}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed at current step: {e}")
        logger.error("Check the logs above to identify which step failed")
        raise


if __name__ == "__main__":
    # If no command is provided, run the full pipeline
    import sys
    if len(sys.argv) == 1:
        full_pipeline()
    else:
        app()
