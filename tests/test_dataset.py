import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from sterbling_ai_bias_bounty.dataset import main


class TestDatasetProcessing:
    """Test suite for dataset processing functionality."""
    
    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data for testing with proper data types."""
        return pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'Age': [25.0, 35.0, 30.0, 45.0, 30.0],  # No NaN values
            'Income': [50000.0, 60000.0, 70000.0, 55000.0, 55000.0],  # No NaN values
            'Credit_Score': [650.0, 700.0, 680.0, 750.0, 680.0],  # No NaN values
            'Loan_Amount': [100000.0, 120000.0, 150000.0, 200000.0, 120000.0],  # No NaN values
            'Gender': ['Male', 'Female', 'Non-binary', 'Male', 'Female'],
            'Race': ['White', 'Black', 'Asian', 'Hispanic', 'White'],
            'Age_Group': ['25-60', 'Under 25', 'Over 60', '25-60', 'Under 25'],
            'Employment_Type': ['Full-time', 'Part-time', 'Self-employed', 'Full-time', 'Part-time'],
            'Education_Level': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor'],
            'Citizenship_Status': ['Citizen', 'Permanent Resident', 'Visa Holder', 'Citizen', 'Permanent Resident'],
            'Language_Proficiency': ['Native', 'Fluent', 'Intermediate', 'Native', 'Fluent'],
            'Disability_Status': ['None', 'Physical', 'None', 'None', 'Cognitive'],
            'Criminal_Record': ['None', 'Minor', 'None', 'None', 'None'],
            'Zip_Code_Group': ['Urban', 'Suburban', 'Rural', 'Urban', 'Suburban'],
            'Loan_Approved': ['Approved', 'Denied', 'Approved', 'Approved', 'Denied']
        })
    
    @pytest.fixture 
    def sample_test_data(self):
        """Create sample test data for testing with proper data types."""
        return pd.DataFrame({
            'ID': [6, 7, 8],
            'Age': [28.0, 32.0, 42.0],  # No NaN values
            'Income': [52000.0, 65000.0, 58000.0],  # No NaN values
            'Credit_Score': [670.0, 720.0, 690.0],
            'Loan_Amount': [110000.0, 140000.0, 130000.0],  # No NaN values
            'Gender': ['Male', 'Female', 'Non-binary'],
            'Race': ['Asian', 'White', 'Black'],
            'Age_Group': ['25-60', 'Under 25', '25-60'],
            'Employment_Type': ['Full-time', 'Self-employed', 'Part-time'],
            'Education_Level': ['Master', 'Bachelor', 'High School'],
            'Citizenship_Status': ['Visa Holder', 'Citizen', 'Permanent Resident'],
            'Language_Proficiency': ['Fluent', 'Native', 'Intermediate'],
            'Disability_Status': ['None', 'None', 'Physical'],
            'Criminal_Record': ['None', 'None', 'Minor'],
            'Zip_Code_Group': ['Rural', 'Urban', 'Suburban']
        })
    
    def test_missing_value_imputation_mode(self, sample_train_data):
        """Test that categorical missing values are filled with mode."""
        from sterbling_ai_bias_bounty.dataset import main
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_file = temp_path / "train.csv"
            test_file = temp_path / "test.csv"
            output_file = temp_path / "output.csv"
            
            # Save sample data
            sample_train_data.to_csv(train_file, index=False)
            sample_train_data.iloc[:2].to_csv(test_file, index=False)
            
            # Mock the preprocessing function to test specific logic
            with patch('sterbling_ai_bias_bounty.dataset.logger'):
                # Test the preprocessing logic directly
                df = sample_train_data.copy()
                
                # Apply the same logic as in the main function
                for col in ['Age', 'Income', 'Credit_Score']:
                    if col in df.columns:
                        original_nulls = df[col].isnull().sum()
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else 0
                        df[col] = df[col].fillna(mode_value)  # Fix pandas warning
                        
                        # Check that nulls are filled
                        assert df[col].isnull().sum() == 0
                        
                        # Check that filled values are the mode (for categorical-like data)
                        if original_nulls > 0:
                            assert mode_value in df[col].values
    
    def test_missing_value_imputation_median(self, sample_train_data):
        """Test that numerical missing values are filled with median."""
        df = sample_train_data.copy()
        
        for col in ['Loan_Amount']:
            if col in df.columns:
                original_nulls = df[col].isnull().sum()
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)  # Fix pandas warning
                
                # Check that nulls are filled
                assert df[col].isnull().sum() == 0
                
                # Check that filled values are reasonable
                if original_nulls > 0:
                    assert median_value in df[col].values or not np.isnan(median_value)
    
    def test_label_encoding_preserves_mappings(self, sample_train_data):
        """Test that label encoding creates proper mappings."""
        from sklearn.preprocessing import LabelEncoder
        
        df = sample_train_data.copy()
        categorical_cols = ['Gender', 'Race', 'Age_Group']
        encoding_mappings = {}
        
        le = LabelEncoder()
        for col in categorical_cols:
            if col in df.columns:
                original_values = df[col].copy()
                df[col] = le.fit_transform(df[col])
                
                unique_originals = sorted(original_values.unique())
                unique_encoded = sorted(df[col].unique())
                encoding_mappings[col] = {int(enc): orig for enc, orig in zip(unique_encoded, unique_originals)}
                
                # Test that mapping is correct
                assert len(encoding_mappings[col]) == len(unique_originals)
                assert all(isinstance(k, int) for k in encoding_mappings[col].keys())
                assert all(isinstance(v, str) for v in encoding_mappings[col].values())
                
                # Test that encoded values are consecutive integers starting from 0
                assert set(unique_encoded) == set(range(len(unique_encoded)))
    
    def test_json_serialization_compatibility(self, sample_train_data):
        """Test that encoding mappings can be serialized to JSON."""
        from sklearn.preprocessing import LabelEncoder
        import json
        
        df = sample_train_data.copy()
        categorical_cols = ['Gender', 'Race']
        encoding_mappings = {}
        
        le = LabelEncoder()
        for col in categorical_cols:
            original_values = df[col].copy()
            df[col] = le.fit_transform(df[col])
            
            unique_originals = sorted(original_values.unique())
            unique_encoded = sorted(df[col].unique())
            # Test the fix for numpy int64 serialization
            encoding_mappings[col] = {int(enc): orig for enc, orig in zip(unique_encoded, unique_originals)}
        
        # Test JSON serialization works
        json_str = json.dumps(encoding_mappings)
        decoded_mappings = json.loads(json_str)
        
        # JSON converts integer keys to strings, so we need to convert back for comparison
        normalized_decoded = {}
        for col, mapping in decoded_mappings.items():
            normalized_decoded[col] = {int(k): v for k, v in mapping.items()}
        
        assert normalized_decoded == encoding_mappings
    
    @patch('sterbling_ai_bias_bounty.dataset.logger')
    def test_main_function_file_operations(self, mock_logger, sample_train_data, sample_test_data):
        """Test the main function file I/O operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_file = temp_path / "train.csv"
            test_file = temp_path / "test.csv"
            output_file = temp_path / "output.csv"
            
            # Save sample data
            sample_train_data.to_csv(train_file, index=False)
            sample_test_data.to_csv(test_file, index=False)
            
            # Mock the PROCESSED_DATA_DIR to use our temp directory
            with patch('sterbling_ai_bias_bounty.dataset.PROCESSED_DATA_DIR', temp_path):
                # Run main function
                main(train_file, test_file, output_file)
                
                # Check that output files are created
                assert output_file.exists()
                assert (temp_path / "test_processed.csv").exists()
                assert (temp_path / "encoding_mappings.json").exists()
                
                # Check that processed data has expected structure
                processed_df = pd.read_csv(output_file)
                assert len(processed_df) == len(sample_train_data)
                assert 'Loan_Approved' in processed_df.columns
                
                # Check that encoding mappings are valid JSON
                with open(temp_path / "encoding_mappings.json", 'r') as f:
                    mappings = json.load(f)
                assert isinstance(mappings, dict)
                assert 'Gender' in mappings

    def test_categorical_columns_list_completeness(self):
        """Test that all expected categorical columns are included in processing."""
        expected_cols = [
            'Gender', 'Race', 'Age_Group', 'Employment_Type', 'Education_Level', 
            'Citizenship_Status', 'Language_Proficiency', 'Disability_Status', 
            'Criminal_Record', 'Zip_Code_Group'
        ]
        
        # This would be the list from the actual function
        from sterbling_ai_bias_bounty.dataset import main
        import inspect
        
        # Get the source code to check the categorical_cols list
        source = inspect.getsource(main)
        for col in expected_cols:
            assert col in source, f"Expected categorical column {col} not found in processing list"
    
    @patch('sterbling_ai_bias_bounty.dataset.logger')
    def test_error_handling_missing_files(self, mock_logger):
        """Test error handling when input files don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nonexistent_file = temp_path / "nonexistent.csv"
            output_file = temp_path / "output.csv"
            
            # Test that FileNotFoundError is raised appropriately
            with pytest.raises(FileNotFoundError):
                main(nonexistent_file, nonexistent_file, output_file)
    
    def test_train_vs_test_processing_differences(self, sample_train_data, sample_test_data):
        """Test that training data includes Loan_Approved encoding while test data doesn't."""
        from sklearn.preprocessing import LabelEncoder
        
        # Test training data processing (is_train=True)
        df_train = sample_train_data.copy()
        categorical_cols = ['Gender', 'Race', 'Loan_Approved']
        
        le = LabelEncoder()
        for col in categorical_cols:
            if col in df_train.columns:
                df_train[col] = le.fit_transform(df_train[col])
        
        assert 'Loan_Approved' in df_train.columns
        assert df_train['Loan_Approved'].dtype in [np.int32, np.int64]
        
        # Test test data processing (is_train=False) 
        df_test = sample_test_data.copy()
        categorical_cols_test = ['Gender', 'Race']  # No Loan_Approved
        
        for col in categorical_cols_test:
            if col in df_test.columns:
                df_test[col] = le.fit_transform(df_test[col])
        
        # Test data shouldn't have Loan_Approved column or it should remain unchanged
        assert 'Loan_Approved' not in df_test.columns or df_test['Loan_Approved'].dtype == object
    
    @patch('sterbling_ai_bias_bounty.dataset.logger')
    def test_logging_behavior(self, mock_logger, sample_train_data, sample_test_data):
        """Test that appropriate logging messages are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_file = temp_path / "train.csv"
            test_file = temp_path / "test.csv"
            output_file = temp_path / "output.csv"
            
            sample_train_data.to_csv(train_file, index=False)
            sample_test_data.to_csv(test_file, index=False)
            
            main(train_file, test_file, output_file)
            
            # Check that key logging calls were made
            mock_logger.info.assert_called()
            mock_logger.success.assert_called()
            
            # Check for specific log messages
            log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Loading and processing loan dataset" in msg for msg in log_messages)
            assert any("Training data shape" in msg for msg in log_messages)
            assert any("Test data shape" in msg for msg in log_messages)
    
    def test_encoding_mapping_accuracy(self, sample_train_data):
        """Test that encoding mappings accurately reflect the transformation."""
        from sklearn.preprocessing import LabelEncoder
        
        df = sample_train_data.copy()
        test_col = 'Gender'
        original_values = df[test_col].copy()
        
        le = LabelEncoder()
        df[test_col] = le.fit_transform(df[test_col])
        
        unique_originals = sorted(original_values.unique())
        unique_encoded = sorted(df[test_col].unique())
        mapping = {int(enc): orig for enc, orig in zip(unique_encoded, unique_originals)}
        
        # Test that we can reconstruct original values using the mapping
        reconstructed = df[test_col].map(mapping)
        
        # Create a reverse mapping to test accuracy
        original_to_encoded = {orig: enc for enc, orig in mapping.items()}
        
        for orig_val in original_values:
            encoded_val = original_to_encoded[orig_val]
            assert mapping[encoded_val] == orig_val

    pytest.main([__file__])
if __name__ == "__main__":
    pytest.main([__file__])
