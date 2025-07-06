import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sterbling_ai_bias_bounty.modeling.predict import main


class TestModelPrediction:
    """Test suite for model prediction functionality."""
    
    @pytest.fixture
    def sample_test_features_with_id(self):
        """Create sample test features data with ID column."""
        np.random.seed(42)
        n_samples = 50
        
        return pd.DataFrame({
            'ID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 80, n_samples),
            'Income': np.random.randint(30000, 150000, n_samples),
            'Credit_Score': np.random.randint(300, 850, n_samples),
            'Loan_Amount': np.random.randint(50000, 500000, n_samples),
            'Gender': np.random.randint(0, 3, n_samples),  # Encoded
            'Race': np.random.randint(0, 5, n_samples),  # Encoded
            'Age_Group': np.random.randint(0, 3, n_samples),  # Encoded
            'Employment_Type': np.random.randint(0, 4, n_samples),  # Encoded
            'Education_Level': np.random.randint(0, 4, n_samples),  # Encoded
            'Citizenship_Status': np.random.randint(0, 3, n_samples),  # Encoded
            'Language_Proficiency': np.random.randint(0, 4, n_samples),  # Encoded
            'Disability_Status': np.random.randint(0, 4, n_samples),  # Encoded
            'Criminal_Record': np.random.randint(0, 3, n_samples),  # Encoded
            'Zip_Code_Group': np.random.randint(0, 3, n_samples)  # Encoded
        })
    
    @pytest.fixture
    def sample_test_features_without_id(self):
        """Create sample test features data without ID column."""
        np.random.seed(42)
        n_samples = 30
        
        return pd.DataFrame({
            'Age': np.random.randint(18, 80, n_samples),
            'Income': np.random.randint(30000, 150000, n_samples),
            'Credit_Score': np.random.randint(300, 850, n_samples),
            'Loan_Amount': np.random.randint(50000, 500000, n_samples),
            'Gender': np.random.randint(0, 3, n_samples),
            'Race': np.random.randint(0, 5, n_samples),
            'Age_Group': np.random.randint(0, 3, n_samples),
            'Employment_Type': np.random.randint(0, 4, n_samples),
            'Education_Level': np.random.randint(0, 4, n_samples),
            'Citizenship_Status': np.random.randint(0, 3, n_samples),
            'Language_Proficiency': np.random.randint(0, 4, n_samples),
            'Disability_Status': np.random.randint(0, 4, n_samples),
            'Criminal_Record': np.random.randint(0, 3, n_samples),
            'Zip_Code_Group': np.random.randint(0, 3, n_samples)
        })
    
    @pytest.fixture
    def trained_random_forest_model(self, sample_test_features_without_id):
        """Create a trained Random Forest model for testing."""
        # Create training data
        X_train = sample_test_features_without_id.copy()
        y_train = np.random.randint(0, 2, len(X_train))
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    @pytest.fixture
    def trained_logistic_regression_model(self, sample_test_features_without_id):
        """Create a trained Logistic Regression model for testing."""
        # Create training data
        X_train = sample_test_features_without_id.copy()
        y_train = np.random.randint(0, 2, len(X_train))
        
        # Train model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        return model, scaler
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_successful_prediction_random_forest(self, mock_logger, sample_test_features_with_id, trained_random_forest_model):
        """Test successful prediction pipeline with Random Forest model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "test_features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            # Save test data and model
            sample_test_features_with_id.to_csv(features_file, index=False)
            joblib.dump(trained_random_forest_model, model_file)
            
            # Mock directory paths
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    # Run prediction
                    main(features_file, model_file, predictions_file)
                    
                    # Check that prediction file is created
                    assert predictions_file.exists()
                    
                    # Load and verify predictions
                    predictions_df = pd.read_csv(predictions_file)
                    
                    # Should have same number of rows as test data
                    assert len(predictions_df) == len(sample_test_features_with_id)
                    
                    # Should have required columns
                    assert 'ID' in predictions_df.columns
                    assert 'Loan_Approved' in predictions_df.columns
                    
                    # Should have valid prediction values
                    valid_values = predictions_df['Loan_Approved'].isin(['Approved', 'Denied'])
                    assert valid_values.all()
                    
                    # Check detailed predictions file
                    detailed_file = temp_path / "detailed_predictions.csv"
                    assert detailed_file.exists()
                    
                    detailed_df = pd.read_csv(detailed_file)
                    assert 'Approval_Probability' in detailed_df.columns
                    assert 'Prediction_Confidence' in detailed_df.columns
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_successful_prediction_logistic_regression_with_scaling(self, mock_logger, sample_test_features_with_id, trained_logistic_regression_model):
        """Test successful prediction with Logistic Regression requiring scaling."""
        model, scaler = trained_logistic_regression_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "test_features.csv"
            model_file = temp_path / "model.pkl"
            scaler_file = temp_path / "scaler.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            # Save test data, model, and scaler
            sample_test_features_with_id.to_csv(features_file, index=False)
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            
            # Mock directory paths
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    # Run prediction
                    main(features_file, model_file, predictions_file)
                    
                    # Check that prediction file is created
                    assert predictions_file.exists()
                    
                    # Verify predictions format
                    predictions_df = pd.read_csv(predictions_file)
                    assert len(predictions_df) == len(sample_test_features_with_id)
                    assert set(predictions_df.columns) == {'ID', 'Loan_Approved'}
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_prediction_without_id_column(self, mock_logger, sample_test_features_without_id, trained_random_forest_model):
        """Test prediction when test data doesn't have ID column."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "test_features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            # Save test data without ID and model
            sample_test_features_without_id.to_csv(features_file, index=False)
            joblib.dump(trained_random_forest_model, model_file)
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    main(features_file, model_file, predictions_file)
                    
                    # Should create predictions with generated IDs
                    predictions_df = pd.read_csv(predictions_file)
                    assert len(predictions_df) == len(sample_test_features_without_id)
                    assert 'ID' in predictions_df.columns
                    
                    # Generated IDs should be sequential
                    expected_ids = list(range(len(sample_test_features_without_id)))
                    assert predictions_df['ID'].tolist() == expected_ids
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_alternative_features_file_path(self, mock_logger, sample_test_features_with_id, trained_random_forest_model):
        """Test that alternative features file path is tried when primary doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nonexistent_features = temp_path / "nonexistent_features.csv"
            alternative_features = temp_path / "test_processed.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            # Save data to alternative path only
            sample_test_features_with_id.to_csv(alternative_features, index=False)
            joblib.dump(trained_random_forest_model, model_file)
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    main(nonexistent_features, model_file, predictions_file)
                    
                    # Should succeed using alternative path
                    assert predictions_file.exists()
    
    def test_error_handling_missing_model_file(self):
        """Test error handling when model file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "features.csv"
            nonexistent_model = temp_path / "nonexistent_model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            # Create features file but not model
            pd.DataFrame({'Age': [25, 30, 35]}).to_csv(features_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    with patch('sterbling_ai_bias_bounty.modeling.predict.logger') as mock_logger:
                        # Should return early without crashing
                        main(features_file, nonexistent_model, predictions_file)
                        
                        # Should not create predictions file
                        assert not predictions_file.exists()
                        
                        # Check that error was logged
                        mock_logger.error.assert_called()
                        error_messages = [call.args[0] for call in mock_logger.error.call_args_list]
                        assert any("Model file not found" in msg for msg in error_messages)
    
    def test_error_handling_missing_features_file(self):
        """Test error handling when features file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nonexistent_features = temp_path / "nonexistent_features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            # Create model file but not features
            joblib.dump(RandomForestClassifier(), model_file)
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    with patch('sterbling_ai_bias_bounty.modeling.predict.logger') as mock_logger:
                        main(nonexistent_features, model_file, predictions_file)
                        
                        # Should not create predictions file
                        assert not predictions_file.exists()
                        
                        # Check that error was logged
                        error_messages = [call.args[0] for call in mock_logger.error.call_args_list]
                        assert any("Test features file not found" in msg for msg in error_messages)
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_prediction_statistics_logging(self, mock_logger, sample_test_features_with_id, trained_random_forest_model):
        """Test that prediction statistics are properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "test_features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            sample_test_features_with_id.to_csv(features_file, index=False)
            joblib.dump(trained_random_forest_model, model_file)
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    main(features_file, model_file, predictions_file)
                    
                    # Check that statistics were logged
                    info_messages = [call.args[0] for call in mock_logger.info.call_args_list]
                    
                    # Should log approval rate
                    approval_rate_messages = [msg for msg in info_messages if "Approval rate" in msg]
                    assert len(approval_rate_messages) >= 1
                    
                    # Should log average approval probability
                    prob_messages = [msg for msg in info_messages if "Average approval probability" in msg]
                    assert len(prob_messages) >= 1
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_submission_format_matches_notebook(self, mock_logger, sample_test_features_with_id, trained_random_forest_model):
        """Test that submission format matches the notebook's submission.csv format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "test_features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            sample_test_features_with_id.to_csv(features_file, index=False)
            joblib.dump(trained_random_forest_model, model_file)
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    main(features_file, model_file, predictions_file)
                    
                    # Load predictions
                    predictions_df = pd.read_csv(predictions_file)
                    
                    # Should match notebook's submission format exactly
                    assert list(predictions_df.columns) == ['ID', 'Loan_Approved']
                    
                    # Should use 'Approved'/'Denied' format from notebook
                    unique_values = set(predictions_df['Loan_Approved'].unique())
                    assert unique_values.issubset({'Approved', 'Denied'})
                    
                    # IDs should match input
                    pd.testing.assert_series_equal(
                        predictions_df['ID'], 
                        sample_test_features_with_id['ID'], 
                        check_names=False
                    )
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_detailed_predictions_structure(self, mock_logger, sample_test_features_with_id, trained_random_forest_model):
        """Test the structure of detailed predictions file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "test_features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            sample_test_features_with_id.to_csv(features_file, index=False)
            joblib.dump(trained_random_forest_model, model_file)
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    main(features_file, model_file, predictions_file)
                    
                    # Check detailed predictions
                    detailed_file = temp_path / "detailed_predictions.csv"
                    detailed_df = pd.read_csv(detailed_file)
                    
                    # Should have all required columns
                    expected_cols = ['ID', 'Loan_Approved', 'Approval_Probability', 'Prediction_Confidence']
                    assert list(detailed_df.columns) == expected_cols
                    
                    # Probabilities should be between 0 and 1
                    assert (detailed_df['Approval_Probability'] >= 0).all()
                    assert (detailed_df['Approval_Probability'] <= 1).all()
                    
                    # Confidence should be between 0.5 and 1
                    assert (detailed_df['Prediction_Confidence'] >= 0.5).all()
                    assert (detailed_df['Prediction_Confidence'] <= 1).all()
    
    @patch('sterbling_ai_bias_bounty.modeling.predict.logger')
    def test_prediction_without_scaler_file(self, mock_logger, sample_test_features_with_id, trained_random_forest_model):
        """Test prediction when scaler file doesn't exist (non-scaling model)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "test_features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            sample_test_features_with_id.to_csv(features_file, index=False)
            joblib.dump(trained_random_forest_model, model_file)
            # Note: Not creating scaler.pkl file
            
            with patch('sterbling_ai_bias_bounty.modeling.predict.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.modeling.predict.PROCESSED_DATA_DIR', temp_path):
                    main(features_file, model_file, predictions_file)
                    
                    # Should still work without scaler
                    assert predictions_file.exists()
                    predictions_df = pd.read_csv(predictions_file)
                    assert len(predictions_df) == len(sample_test_features_with_id)


if __name__ == "__main__":
    pytest.main([__file__])
