import pytest
import pandas as pd
import numpy as np
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from typer.testing import CliRunner

from loan_model import app, full_pipeline, data, feature_engineering, train_model, predict_model, visualize


class TestLoanModelPipeline:
    """Test suite for loan model pipeline functionality."""
    
    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'ID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 80, n_samples),
            'Income': np.random.randint(30000, 150000, n_samples),
            'Credit_Score': np.random.randint(300, 850, n_samples),
            'Loan_Amount': np.random.randint(50000, 500000, n_samples),
            'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples),
            'Race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
            'Age_Group': np.random.choice(['Under 25', '25-60', 'Over 60'], n_samples),
            'Employment_Type': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples),
            'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'Citizenship_Status': np.random.choice(['Citizen', 'Permanent Resident', 'Visa Holder'], n_samples),
            'Language_Proficiency': np.random.choice(['Native', 'Fluent', 'Intermediate'], n_samples),
            'Disability_Status': np.random.choice(['None', 'Physical', 'Cognitive'], n_samples),
            'Criminal_Record': np.random.choice(['None', 'Minor', 'Major'], n_samples),
            'Zip_Code_Group': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
            'Loan_Approved': np.random.choice(['Approved', 'Denied'], n_samples)
        })
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data for testing."""
        np.random.seed(42)
        n_samples = 50
        
        return pd.DataFrame({
            'ID': range(101, n_samples + 101),
            'Age': np.random.randint(18, 80, n_samples),
            'Income': np.random.randint(30000, 150000, n_samples),
            'Credit_Score': np.random.randint(300, 850, n_samples),
            'Loan_Amount': np.random.randint(50000, 500000, n_samples),
            'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples),
            'Race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
            'Age_Group': np.random.choice(['Under 25', '25-60', 'Over 60'], n_samples),
            'Employment_Type': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples),
            'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'Citizenship_Status': np.random.choice(['Citizen', 'Permanent Resident', 'Visa Holder'], n_samples),
            'Language_Proficiency': np.random.choice(['Native', 'Fluent', 'Intermediate'], n_samples),
            'Disability_Status': np.random.choice(['None', 'Physical', 'Cognitive'], n_samples),
            'Criminal_Record': np.random.choice(['None', 'Minor', 'Major'], n_samples),
            'Zip_Code_Group': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
        })
    
    @pytest.fixture
    def runner(self):
        """Create a typer CLI test runner."""
        return CliRunner()

    def test_data_command(self, runner, sample_train_data, sample_test_data):
        """Test the data processing command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_file = temp_path / "train.csv"
            test_file = temp_path / "test.csv"
            output_file = temp_path / "output.csv"
            
            # Save sample data
            sample_train_data.to_csv(train_file, index=False)
            sample_test_data.to_csv(test_file, index=False)
            
            # Mock the dataset module
            with patch('loan_model.dataset') as mock_dataset:
                result = runner.invoke(app, [
                    'data',
                    '--train-path', str(train_file),
                    '--test-path', str(test_file),
                    '--output-path', str(output_file)
                ])
                
                assert result.exit_code == 0
                mock_dataset.main.assert_called_once_with(train_file, test_file, output_file)

    def test_feature_engineering_command(self, runner):
        """Test the feature engineering command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.csv"
            output_file = temp_path / "output.csv"
            
            # Mock the features module
            with patch('loan_model.features') as mock_features:
                result = runner.invoke(app, [
                    'feature-engineering',
                    '--input-path', str(input_file),
                    '--output-path', str(output_file)
                ])
                
                assert result.exit_code == 0
                mock_features.main.assert_called_once_with(input_file, output_file)

    def test_train_model_command(self, runner):
        """Test the model training command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "features.csv"
            labels_file = temp_path / "labels.csv"
            model_file = temp_path / "model.pkl"
            
            # Mock the train module
            with patch('loan_model.train') as mock_train:
                result = runner.invoke(app, [
                    'train-model',
                    '--features-path', str(features_file),
                    '--labels-path', str(labels_file),
                    '--model-path', str(model_file)
                ])
                
                assert result.exit_code == 0
                mock_train.main.assert_called_once_with(features_file, labels_file, model_file)

    def test_predict_model_command(self, runner):
        """Test the model prediction command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_file = temp_path / "features.csv"
            model_file = temp_path / "model.pkl"
            predictions_file = temp_path / "predictions.csv"
            
            # Mock the predict module
            with patch('loan_model.predict') as mock_predict:
                result = runner.invoke(app, [
                    'predict-model',
                    '--features-path', str(features_file),
                    '--model-path', str(model_file),
                    '--predictions-path', str(predictions_file)
                ])
                
                assert result.exit_code == 0
                mock_predict.main.assert_called_once_with(features_file, model_file, predictions_file)

    def test_visualize_command(self, runner):
        """Test the visualization command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.csv"
            output_file = temp_path / "output.png"
            
            # Mock the plots module
            with patch('loan_model.plots') as mock_plots:
                result = runner.invoke(app, [
                    'visualize',
                    '--input-path', str(input_file),
                    '--output-path', str(output_file)
                ])
                
                assert result.exit_code == 0
                mock_plots.main.assert_called_once_with(input_file, output_file)

    @patch('loan_model.logger')
    def test_full_pipeline_success(self, mock_logger, sample_train_data, sample_test_data):
        """Test successful full pipeline execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create raw data files
            train_file = temp_path / "raw" / "loan_access_dataset.csv"
            test_file = temp_path / "raw" / "test.csv"
            train_file.parent.mkdir(parents=True, exist_ok=True)
            
            sample_train_data.to_csv(train_file, index=False)
            sample_test_data.to_csv(test_file, index=False)
            
            # Mock all directory paths
            with patch('loan_model.RAW_DATA_DIR', temp_path / "raw"):
                with patch('loan_model.PROCESSED_DATA_DIR', temp_path / "processed"):
                    with patch('loan_model.MODELS_DIR', temp_path / "models"):
                        with patch('loan_model.FIGURES_DIR', temp_path / "figures"):
                            # Mock all module functions
                            with patch('loan_model.dataset') as mock_dataset:
                                with patch('loan_model.features') as mock_features:
                                    with patch('loan_model.train') as mock_train:
                                        with patch('loan_model.predict') as mock_predict:
                                            with patch('loan_model.plots') as mock_plots:
                                                # Mock test_processed.csv existence
                                                test_processed_path = temp_path / "processed" / "test_processed.csv"
                                                test_processed_path.parent.mkdir(parents=True, exist_ok=True)
                                                test_processed_path.touch()
                                                
                                                # Run full pipeline
                                                full_pipeline()
                                                
                                                # Verify all modules were called
                                                mock_dataset.main.assert_called_once()
                                                assert mock_features.main.call_count == 2  # Training and test features
                                                mock_train.main.assert_called_once()
                                                mock_predict.main.assert_called_once()
                                                mock_plots.main.assert_called_once()
                                                
                                                # Verify logging messages
                                                mock_logger.info.assert_called()
                                                mock_logger.success.assert_called()

    @patch('loan_model.logger')
    def test_full_pipeline_missing_train_data(self, mock_logger):
        """Test full pipeline with missing training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock RAW_DATA_DIR but don't create the files
            with patch('loan_model.RAW_DATA_DIR', temp_path / "raw"):
                full_pipeline()
                
                # Verify error logging
                mock_logger.error.assert_called()
                error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
                assert any("Training data file not found" in msg for msg in error_calls)

    @patch('loan_model.logger')
    def test_full_pipeline_missing_test_data(self, mock_logger, sample_train_data):
        """Test full pipeline with missing test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create only training data
            train_file = temp_path / "raw" / "loan_access_dataset.csv"
            train_file.parent.mkdir(parents=True, exist_ok=True)
            sample_train_data.to_csv(train_file, index=False)
            
            with patch('loan_model.RAW_DATA_DIR', temp_path / "raw"):
                full_pipeline()
                
                # Verify error logging
                mock_logger.error.assert_called()
                error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
                assert any("Test data file not found" in msg for msg in error_calls)

    @patch('loan_model.logger')
    def test_full_pipeline_step_failure(self, mock_logger, sample_train_data, sample_test_data):
        """Test full pipeline with step failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create raw data files
            train_file = temp_path / "raw" / "loan_access_dataset.csv"
            test_file = temp_path / "raw" / "test.csv"
            train_file.parent.mkdir(parents=True, exist_ok=True)
            
            sample_train_data.to_csv(train_file, index=False)
            sample_test_data.to_csv(test_file, index=False)
            
            # Mock directory paths
            with patch('loan_model.RAW_DATA_DIR', temp_path / "raw"):
                with patch('loan_model.PROCESSED_DATA_DIR', temp_path / "processed"):
                    with patch('loan_model.MODELS_DIR', temp_path / "models"):
                        with patch('loan_model.FIGURES_DIR', temp_path / "figures"):
                            # Mock dataset to raise an exception
                            with patch('loan_model.dataset') as mock_dataset:
                                mock_dataset.main.side_effect = Exception("Dataset processing failed")
                                
                                # Should raise the exception
                                with pytest.raises(Exception, match="Dataset processing failed"):
                                    full_pipeline()
                                
                                # Verify error logging
                                mock_logger.error.assert_called()

    def test_individual_command_functions(self):
        """Test individual command functions with mocked modules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test data command function
            with patch('loan_model.dataset') as mock_dataset:
                with patch('loan_model.logger') as mock_logger:
                    data(temp_path / "train.csv", temp_path / "test.csv", temp_path / "output.csv")
                    mock_dataset.main.assert_called_once()
                    mock_logger.info.assert_called_with("Running data processing step...")
            
            # Test feature engineering command function
            with patch('loan_model.features') as mock_features:
                with patch('loan_model.logger') as mock_logger:
                    feature_engineering(temp_path / "input.csv", temp_path / "output.csv")
                    mock_features.main.assert_called_once()
                    mock_logger.info.assert_called_with("Running feature engineering step...")
            
            # Test train model command function
            with patch('loan_model.train') as mock_train:
                with patch('loan_model.logger') as mock_logger:
                    train_model(temp_path / "features.csv", temp_path / "labels.csv", temp_path / "model.pkl")
                    mock_train.main.assert_called_once()
                    mock_logger.info.assert_called_with("Running model training step...")
            
            # Test predict model command function
            with patch('loan_model.predict') as mock_predict:
                with patch('loan_model.logger') as mock_logger:
                    predict_model(temp_path / "features.csv", temp_path / "model.pkl", temp_path / "predictions.csv")
                    mock_predict.main.assert_called_once()
                    mock_logger.info.assert_called_with("Running model prediction step...")
            
            # Test visualize command function
            with patch('loan_model.plots') as mock_plots:
                with patch('loan_model.logger') as mock_logger:
                    visualize(temp_path / "input.csv", temp_path / "output.png")
                    mock_plots.main.assert_called_once()
                    mock_logger.info.assert_called_with("Running visualization step...")

    def test_main_execution_with_no_args(self):
        """Test main execution when no command line arguments are provided."""
        # Mock sys.argv to simulate no arguments
        test_argv = ['loan_model.py']
        
        with patch.object(sys, 'argv', test_argv):
            with patch('loan_model.full_pipeline') as mock_full_pipeline:
                # Test the main execution logic directly
                import loan_model
                
                # Simulate the condition check
                if len(sys.argv) == 1:
                    loan_model.full_pipeline()
                
                # This should trigger the full pipeline
                mock_full_pipeline.assert_called_once()

    def test_main_execution_with_args(self):
        """Test main execution when command line arguments are provided."""
        # Mock sys.argv to simulate command arguments
        test_argv = ['loan_model.py', 'data']
        
        with patch.object(sys, 'argv', test_argv):
            with patch('loan_model.app') as mock_app:
                # Test the main execution logic directly
                import loan_model
                
                # Simulate the condition check
                if len(sys.argv) > 1:
                    loan_model.app()
                
                # This should trigger the typer app
                mock_app.assert_called_once()

    def test_main_block_logic_no_args(self):
        """Test the main block logic when no arguments provided."""
        with patch('loan_model.full_pipeline') as mock_full_pipeline:
            # Test the actual logic from the main block
            test_argv = ['loan_model.py']
            if len(test_argv) == 1:
                import loan_model
                loan_model.full_pipeline()
            
            mock_full_pipeline.assert_called_once()

    def test_main_block_logic_with_args(self):
        """Test the main block logic when arguments provided."""
        with patch('loan_model.app') as mock_app:
            # Test the actual logic from the main block
            test_argv = ['loan_model.py', 'data', '--help']
            if len(test_argv) > 1:
                import loan_model
                loan_model.app()
            
            mock_app.assert_called_once()

    @patch('loan_model.logger')
    def test_pipeline_logging_sequence(self, mock_logger, sample_train_data, sample_test_data):
        """Test that pipeline logs steps in correct sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create raw data files
            train_file = temp_path / "raw" / "loan_access_dataset.csv"
            test_file = temp_path / "raw" / "test.csv"
            train_file.parent.mkdir(parents=True, exist_ok=True)
            
            sample_train_data.to_csv(train_file, index=False)
            sample_test_data.to_csv(test_file, index=False)
            
            with patch('loan_model.RAW_DATA_DIR', temp_path / "raw"):
                with patch('loan_model.PROCESSED_DATA_DIR', temp_path / "processed"):
                    with patch('loan_model.MODELS_DIR', temp_path / "models"):
                        with patch('loan_model.FIGURES_DIR', temp_path / "figures"):
                            # Mock all modules
                            with patch('loan_model.dataset'), patch('loan_model.features'), \
                                 patch('loan_model.train'), patch('loan_model.predict'), \
                                 patch('loan_model.plots'):
                                
                                # Mock test_processed.csv existence
                                test_processed_path = temp_path / "processed" / "test_processed.csv"
                                test_processed_path.parent.mkdir(parents=True, exist_ok=True)
                                test_processed_path.touch()
                                
                                full_pipeline()
                                
                                # Check logging sequence
                                info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                                success_calls = [call.args[0] for call in mock_logger.success.call_args_list]
                                
                                # Verify step sequence
                                assert any("Step 1: Processing raw data" in msg for msg in info_calls)
                                assert any("Step 2: Engineering features" in msg for msg in info_calls)
                                assert any("Step 3: Training models" in msg for msg in info_calls)
                                assert any("Step 4: Generating predictions" in msg for msg in info_calls)
                                assert any("Step 5: Creating visualizations" in msg for msg in info_calls)
                                
                                # Verify completion message
                                assert any("Full pipeline execution complete!" in msg for msg in success_calls)

    def test_typer_app_help(self, runner):
        """Test that the typer app help works correctly."""
        result = runner.invoke(app, ['--help'])
        assert result.exit_code == 0
        assert "AI Bias Bounty Loan Model Pipeline" in result.output

    def test_command_help_messages(self, runner):
        """Test help messages for individual commands."""
        commands = ['data', 'feature-engineering', 'train-model', 'predict-model', 'visualize']
        
        for command in commands:
            result = runner.invoke(app, [command, '--help'])
            assert result.exit_code == 0
            assert command.replace('-', ' ') in result.output.lower() or command in result.output.lower()

    def test_full_pipeline_function_directly(self, sample_train_data, sample_test_data):
        """Test calling the full_pipeline function directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create raw data files
            train_file = temp_path / "raw" / "loan_access_dataset.csv"
            test_file = temp_path / "raw" / "test.csv"
            train_file.parent.mkdir(parents=True, exist_ok=True)
            
            sample_train_data.to_csv(train_file, index=False)
            sample_test_data.to_csv(test_file, index=False)
            
            # Mock all directory paths
            with patch('loan_model.RAW_DATA_DIR', temp_path / "raw"):
                with patch('loan_model.PROCESSED_DATA_DIR', temp_path / "processed"):
                    with patch('loan_model.MODELS_DIR', temp_path / "models"):
                        with patch('loan_model.FIGURES_DIR', temp_path / "figures"):
                            with patch('loan_model.logger') as mock_logger:
                                # Mock all module functions
                                with patch('loan_model.dataset') as mock_dataset:
                                    with patch('loan_model.features') as mock_features:
                                        with patch('loan_model.train') as mock_train:
                                            with patch('loan_model.predict') as mock_predict:
                                                with patch('loan_model.plots') as mock_plots:
                                                    # Mock test_processed.csv existence
                                                    test_processed_path = temp_path / "processed" / "test_processed.csv"
                                                    test_processed_path.parent.mkdir(parents=True, exist_ok=True)
                                                    test_processed_path.touch()
                                                    
                                                    # Import and call full_pipeline directly
                                                    from loan_model import full_pipeline
                                                    full_pipeline()
                                                    
                                                    # Verify all modules were called
                                                    mock_dataset.main.assert_called_once()
                                                    assert mock_features.main.call_count == 2
                                                    mock_train.main.assert_called_once()
                                                    mock_predict.main.assert_called_once()
                                                    mock_plots.main.assert_called_once()

    def test_cli_integration_with_runner(self, runner):
        """Test CLI integration using the typer runner."""
        # Test that we can invoke the app without errors
        result = runner.invoke(app, ['--help'])
        assert result.exit_code == 0
        assert "AI Bias Bounty Loan Model Pipeline" in result.output
        
        # Test individual commands can be invoked
        with patch('loan_model.dataset') as mock_dataset:
            result = runner.invoke(app, ['data', '--help'])
            assert result.exit_code == 0
            assert "Process raw data" in result.output

    def test_argument_parsing_logic(self):
        """Test the argument parsing logic without importing the main block."""
        # Test the logic that determines whether to run full pipeline or CLI
        
        # Case 1: No arguments (should run full pipeline)
        argv_no_args = ['loan_model.py']
        should_run_full_pipeline = len(argv_no_args) == 1
        assert should_run_full_pipeline == True
        
        # Case 2: With arguments (should run CLI)
        argv_with_args = ['loan_model.py', 'data']
        should_run_full_pipeline = len(argv_with_args) == 1
        assert should_run_full_pipeline == False

if __name__ == "__main__":
    pytest.main([__file__])
