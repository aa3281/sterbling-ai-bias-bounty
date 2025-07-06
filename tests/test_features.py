import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from sterbling_ai_bias_bounty.features import main


class TestFeaturesProcessing:
    """Test suite for features processing functionality."""
    
    @pytest.fixture
    def sample_processed_data_with_target(self):
        """Create sample processed data with target for training."""
        return pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'Age': [25, 35, 30, 45, 30],
            'Income': [50000, 60000, 70000, 55000, 55000],
            'Credit_Score': [650, 700, 680, 750, 680],
            'Loan_Amount': [100000, 120000, 150000, 200000, 120000],
            'Gender': [1, 0, 2, 1, 0],  # Encoded
            'Race': [4, 1, 0, 2, 4],  # Encoded
            'Age_Group': [1, 2, 0, 1, 2],  # Encoded
            'Employment_Type': [0, 1, 2, 0, 1],  # Encoded
            'Education_Level': [0, 1, 2, 3, 0],  # Encoded
            'Citizenship_Status': [0, 1, 2, 0, 1],  # Encoded
            'Language_Proficiency': [2, 1, 0, 2, 1],  # Encoded
            'Disability_Status': [2, 1, 2, 2, 0],  # Encoded
            'Criminal_Record': [0, 1, 0, 0, 0],  # Encoded
            'Zip_Code_Group': [2, 1, 0, 2, 1],  # Encoded
            'Loan_Approved': [1, 0, 1, 1, 0]  # Target
        })
    
    @pytest.fixture
    def sample_processed_data_without_target(self):
        """Create sample processed data without target for testing."""
        return pd.DataFrame({
            'ID': [6, 7, 8],
            'Age': [28, 32, 42],
            'Income': [52000, 65000, 58000],
            'Credit_Score': [670, 720, 690],
            'Loan_Amount': [110000, 140000, 130000],
            'Gender': [1, 0, 2],  # Encoded
            'Race': [0, 4, 1],  # Encoded
            'Age_Group': [1, 2, 1],  # Encoded
            'Employment_Type': [0, 2, 1],  # Encoded
            'Education_Level': [1, 0, 2],  # Encoded
            'Citizenship_Status': [2, 0, 1],  # Encoded
            'Language_Proficiency': [1, 2, 0],  # Encoded
            'Disability_Status': [2, 2, 1],  # Encoded
            'Criminal_Record': [0, 0, 1],  # Encoded
            'Zip_Code_Group': [0, 2, 1]  # Encoded
        })
    
    def test_feature_separation_with_target(self, sample_processed_data_with_target):
        """Test that features and target are properly separated for training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            # Save sample data
            sample_processed_data_with_target.to_csv(input_file, index=False)
            
            # Mock PROCESSED_DATA_DIR to use temp directory
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    # Run main function
                    main(input_file, features_file)
                    
                    # Check that features file is created
                    assert features_file.exists()
                    
                    # Check that labels file is created
                    labels_file = temp_path / "labels.csv"
                    assert labels_file.exists()
                    
                    # Load and verify features
                    features_df = pd.read_csv(features_file)
                    labels_df = pd.read_csv(labels_file)
                    
                    # Should have 14 features (excluding ID and Loan_Approved)
                    expected_features = 14
                    assert len(features_df.columns) == expected_features
                    
                    # Should not contain ID or target
                    assert 'ID' not in features_df.columns
                    assert 'Loan_Approved' not in features_df.columns
                    
                    # Labels should contain target values
                    assert len(labels_df) == len(sample_processed_data_with_target)
                    assert 'Loan_Approved' in labels_df.columns or labels_df.columns[0] == 'Loan_Approved'
    
    def test_feature_processing_without_target(self, sample_processed_data_without_target):
        """Test that test data is processed correctly without target separation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            # Save sample data
            sample_processed_data_without_target.to_csv(input_file, index=False)
            
            # Mock PROCESSED_DATA_DIR to use temp directory
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    # Run main function
                    main(input_file, features_file)
                    
                    # Check that features file is created
                    assert features_file.exists()
                    
                    # Check that labels file is NOT created
                    labels_file = temp_path / "labels.csv"
                    assert not labels_file.exists()
                    
                    # Load and verify features
                    features_df = pd.read_csv(features_file)
                    
                    # Should exclude only ID, keep all other features
                    expected_features = len(sample_processed_data_without_target.columns) - 1  # minus ID
                    assert len(features_df.columns) == expected_features
                    
                    # Should not contain ID
                    assert 'ID' not in features_df.columns
                    
                    # Should have same number of rows
                    assert len(features_df) == len(sample_processed_data_without_target)
    
    def test_feature_columns_consistency(self, sample_processed_data_with_target):
        """Test that expected feature columns are preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            sample_processed_data_with_target.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    main(input_file, features_file)
                    
                    features_df = pd.read_csv(features_file)
                    
                    # Expected feature columns (from notebook analysis)
                    expected_columns = [
                        'Age', 'Income', 'Credit_Score', 'Loan_Amount',
                        'Gender', 'Race', 'Age_Group', 'Employment_Type',
                        'Education_Level', 'Citizenship_Status', 'Language_Proficiency',
                        'Disability_Status', 'Criminal_Record', 'Zip_Code_Group'
                    ]
                    
                    # Check that all expected columns are present
                    for col in expected_columns:
                        assert col in features_df.columns, f"Expected column {col} not found"
                    
                    # Check that we have exactly the expected number of columns
                    assert len(features_df.columns) == len(expected_columns)
    
    def test_no_additional_feature_engineering(self, sample_processed_data_with_target):
        """Test that no additional feature engineering is performed (matching notebook)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            sample_processed_data_with_target.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    main(input_file, features_file)
                    
                    features_df = pd.read_csv(features_file)
                    
                    # Check that no engineered features are created
                    engineered_features = [
                        'Loan_Amount_Log', 'Income_Log', 'Income_Credit_Ratio',
                        'Loan_Income_Ratio', 'Age_Squared', 'Age_Binned',
                        'Credit_Score_Squared', 'Credit_Score_Binned',
                        'Income_Squared', 'High_Income', 'Large_Loan',
                        'Gender_Income_Interaction', 'Race_Credit_Interaction'
                    ]
                    
                    for eng_feature in engineered_features:
                        assert eng_feature not in features_df.columns, f"Unexpected engineered feature {eng_feature} found"
    
    def test_data_preservation_during_processing(self, sample_processed_data_with_target):
        """Test that original data values are preserved during feature processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            sample_processed_data_with_target.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    main(input_file, features_file)
                    
                    features_df = pd.read_csv(features_file)
                    
                    # Check that feature values match original (excluding ID and target)
                    original_features = sample_processed_data_with_target.drop(['ID', 'Loan_Approved'], axis=1)
                    
                    # Compare shapes
                    assert features_df.shape == original_features.shape
                    
                    # Compare values for key columns
                    for col in ['Age', 'Income', 'Credit_Score', 'Gender']:
                        if col in features_df.columns:
                            pd.testing.assert_series_equal(
                                features_df[col], 
                                original_features[col], 
                                check_names=False
                            )
    
    @patch('sterbling_ai_bias_bounty.features.logger')
    def test_logging_behavior(self, mock_logger, sample_processed_data_with_target):
        """Test that appropriate logging messages are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            sample_processed_data_with_target.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                main(input_file, features_file)
                
                # Check that key logging calls were made
                mock_logger.info.assert_called()
                mock_logger.success.assert_called()
                
                # Check for specific log messages
                log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
                assert any("Generating features from dataset" in msg for msg in log_messages)
                assert any("Dataset shape" in msg for msg in log_messages)
    
    def test_error_handling_missing_input_file(self):
        """Test error handling when input file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nonexistent_file = temp_path / "nonexistent.csv"
            features_file = temp_path / "features.csv"
            
            # Test that FileNotFoundError is raised
            with pytest.raises(FileNotFoundError):
                main(nonexistent_file, features_file)
    
    def test_feature_dtypes_preservation(self, sample_processed_data_with_target):
        """Test that data types are preserved during processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            sample_processed_data_with_target.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    main(input_file, features_file)
                    
                    features_df = pd.read_csv(features_file)
                    
                    # Check that numerical columns have appropriate dtypes
                    numerical_cols = ['Age', 'Income', 'Credit_Score', 'Loan_Amount']
                    for col in numerical_cols:
                        if col in features_df.columns:
                            assert pd.api.types.is_numeric_dtype(features_df[col])
                    
                    # Check that encoded categorical columns are integers
                    categorical_cols = ['Gender', 'Race', 'Age_Group', 'Employment_Type']
                    for col in categorical_cols:
                        if col in features_df.columns:
                            assert pd.api.types.is_integer_dtype(features_df[col])
    
    def test_notebook_compatibility_14_features(self, sample_processed_data_with_target):
        """Test that we match the notebook's 14-feature structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            features_file = temp_path / "features.csv"
            
            sample_processed_data_with_target.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    main(input_file, features_file)
                    
                    features_df = pd.read_csv(features_file)
                    
                    # Notebook has 14 features (X_train shape from Cell 6)
                    assert len(features_df.columns) == 14, f"Expected 14 features to match notebook, got {len(features_df.columns)}"
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty input dataframe."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "empty.csv"
            features_file = temp_path / "features.csv"
            
            # Create empty CSV with headers
            empty_df = pd.DataFrame(columns=['ID', 'Age', 'Income', 'Loan_Approved'])
            empty_df.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.features.PROCESSED_DATA_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.features.logger'):
                    main(input_file, features_file)
                    
                    # Should create empty features file
                    assert features_file.exists()
                    features_df = pd.read_csv(features_file)
                    assert len(features_df) == 0


if __name__ == "__main__":
    pytest.main([__file__])
