import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from sterbling_ai_bias_bounty.plots import main, generate_bias_plots, generate_eda_plots, generate_model_evaluation_plots


class TestPlotsGeneration:
    """Test suite for plots generation functionality."""
    
    def setup_method(self):
        """Setup method to close any existing matplotlib figures."""
        plt.close('all')
    
    def teardown_method(self):
        """Cleanup method to close matplotlib figures after each test."""
        plt.close('all')
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for testing."""
        np.random.seed(42)
        n_samples = 200
        
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
            'Zip_Code_Group': np.random.randint(0, 3, n_samples),  # Encoded
            'Loan_Approved': np.random.randint(0, 2, n_samples)  # Target
        })
    
    @pytest.fixture
    def sample_encoding_mappings(self):
        """Create sample encoding mappings for testing."""
        return {
            'Gender': {0: 'Female', 1: 'Male', 2: 'Non-binary'},
            'Race': {0: 'Asian', 1: 'Black', 2: 'Hispanic', 3: 'Multiracial', 4: 'White'},
            'Age_Group': {0: '25-60', 1: 'Over 60', 2: 'Under 25'},
            'Citizenship_Status': {0: 'Citizen', 1: 'Permanent Resident', 2: 'Visa Holder'}
        }
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 14)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_main_plots_generation_pipeline(self, sample_processed_data):
        """Test the main plots generation pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            output_file = temp_path / "analysis_plots.png"
            
            # Save sample data
            sample_processed_data.to_csv(input_file, index=False)
            
            # Mock directory paths at all levels
            with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                    # Mock the subfunctions to avoid complex dependencies
                    with patch('sterbling_ai_bias_bounty.plots.generate_bias_plots') as mock_bias:
                        with patch('sterbling_ai_bias_bounty.plots.generate_model_interpretability_plots') as mock_interp:
                            with patch('sterbling_ai_bias_bounty.plots.generate_eda_plots') as mock_eda:
                                with patch('sterbling_ai_bias_bounty.plots.generate_model_evaluation_plots') as mock_eval:
                                    # Run main function
                                    main(input_file, output_file)
                                    
                                    # Check that main analysis plot is created
                                    assert output_file.exists()
                                    
                                    # Verify subfunctions were called
                                    mock_bias.assert_called_once()
                                    mock_interp.assert_called_once()
                                    mock_eda.assert_called_once()
                                    mock_eval.assert_called_once()

    def test_bias_plots_generation(self, sample_processed_data, sample_encoding_mappings):
        """Test bias analysis plots generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bias_output = temp_path / "bias_analysis.png"
            mappings_file = temp_path / "encoding_mappings.json"
            
            # Save encoding mappings
            with open(mappings_file, 'w') as f:
                json.dump(sample_encoding_mappings, f)
            
            # Mock directory paths - patch in the config module
            with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                # Generate bias plots
                generate_bias_plots(sample_processed_data, bias_output)
                
                # Check that bias plot is created
                assert bias_output.exists()

    def test_bias_plots_with_missing_encoding_mappings(self, sample_processed_data):
        """Test bias plots generation when encoding mappings don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bias_output = temp_path / "bias_analysis.png"
            
            # Mock directory paths (no mappings file) - patch in the config module
            with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                # Should still work without mappings
                generate_bias_plots(sample_processed_data, bias_output)
                assert bias_output.exists()

    def test_eda_plots_generation(self, sample_processed_data, sample_encoding_mappings):
        """Test EDA plots generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mappings_file = temp_path / "encoding_mappings.json"
            
            # Save encoding mappings
            with open(mappings_file, 'w') as f:
                json.dump(sample_encoding_mappings, f)
            
            # Mock directory paths at all levels
            with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                    # Generate EDA plots
                    generate_eda_plots(sample_processed_data)
                    
                    # Check that EDA plots are created
                    assert (temp_path / "target_distribution.png").exists()
                    assert (temp_path / "numerical_distributions.png").exists()

    def test_eda_plots_numerical_features_filtering(self, sample_processed_data):
        """Test that numerical distributions only include true numerical features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                    generate_eda_plots(sample_processed_data)
                    
                    # Should create numerical distributions plot
                    assert (temp_path / "numerical_distributions.png").exists()

    def test_model_evaluation_plots_with_trained_models(self, trained_model):
        """Test model evaluation plots generation with trained models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            all_models_file = temp_path / "all_models.pkl"
            
            # Create mock validation data
            X_val = pd.DataFrame(np.random.randn(50, 14))
            y_val = np.random.randint(0, 2, 50)
            
            # Create mock models data structure
            models_data = {
                'models': {
                    'RandomForest': {
                        'model': trained_model,
                        'auc': 0.85,
                        'requires_scaling': False
                    }
                },
                'validation_data': (X_val, y_val),
                'scaler': None
            }
            
            # Save models data
            joblib.dump(models_data, all_models_file)
            
            # Mock directory paths at all levels
            with patch('sterbling_ai_bias_bounty.config.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                    # Generate model evaluation plots
                    generate_model_evaluation_plots()
                    
                    # Check that evaluation plots are created
                    assert (temp_path / "confusion_matrix.png").exists()
                    assert (temp_path / "roc_comparison.png").exists()
                    assert (temp_path / "model_summary_table.png").exists()

    def test_model_evaluation_plots_missing_models(self):
        """Test model evaluation plots when no trained models exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock directory paths (no models file)
            with patch('sterbling_ai_bias_bounty.config.MODELS_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                    # Should not crash when no models exist
                    generate_model_evaluation_plots()
                    
                    # Should not create any files
                    assert not (temp_path / "confusion_matrix.png").exists()

    def test_data_shape_logging(self, sample_processed_data):
        """Test that data shape is properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "dataset.csv"
            output_file = temp_path / "analysis_plots.png"
            
            sample_processed_data.to_csv(input_file, index=False)
            
            with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                    # Mock subfunctions to focus on main function
                    with patch('sterbling_ai_bias_bounty.plots.generate_bias_plots'):
                        with patch('sterbling_ai_bias_bounty.plots.generate_model_interpretability_plots'):
                            with patch('sterbling_ai_bias_bounty.plots.generate_eda_plots'):
                                with patch('sterbling_ai_bias_bounty.plots.generate_model_evaluation_plots'):
                                    with patch('builtins.print') as mock_print:
                                        main(input_file, output_file)
                                        
                                        # Check that shape was printed
                                        print_calls = [call.args[0] for call in mock_print.call_args_list]
                                        shape_messages = [msg for msg in print_calls if "Dataset shape" in msg]
                                        assert len(shape_messages) >= 1

    def test_missing_input_file_handling(self):
        """Test handling of missing input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nonexistent_input = temp_path / "nonexistent.csv"
            output_file = temp_path / "analysis_plots.png"
            
            with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                    # Should raise FileNotFoundError
                    with pytest.raises(FileNotFoundError):
                        main(nonexistent_input, output_file)

    def test_figures_directory_creation(self, sample_processed_data):
        """Test that figures directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            figures_path = temp_path / "figures"
            input_file = temp_path / "dataset.csv"
            output_file = figures_path / "analysis_plots.png"
            
            sample_processed_data.to_csv(input_file, index=False)
            
            # figures directory doesn't exist initially
            assert not figures_path.exists()
            
            # Mock directory paths at all levels
            with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', figures_path):
                with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                    # Mock subfunctions
                    with patch('sterbling_ai_bias_bounty.plots.generate_bias_plots'):
                        with patch('sterbling_ai_bias_bounty.plots.generate_model_interpretability_plots'):
                            with patch('sterbling_ai_bias_bounty.plots.generate_eda_plots'):
                                with patch('sterbling_ai_bias_bounty.plots.generate_model_evaluation_plots'):
                                    main(input_file, output_file)
                                    
                                    # Directory should be created
                                    assert figures_path.exists()

    def test_bias_analysis_statistics_calculation(self, sample_processed_data):
        """Test that bias analysis calculates proper statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bias_output = temp_path / "bias_analysis.png"
            
            with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                with patch('builtins.print') as mock_print:
                    generate_bias_plots(sample_processed_data, bias_output)
                    
                    # Check that bias statistics were printed
                    print_calls = [call.args[0] for call in mock_print.call_args_list]
                    
                    # Should print analysis for protected attributes
                    analysis_messages = [msg for msg in print_calls if "Analysis:" in msg]
                    assert len(analysis_messages) >= 1
                    
                    # Should print overall summary
                    summary_messages = [msg for msg in print_calls if "OVERALL BIAS ANALYSIS" in msg]
                    assert len(summary_messages) >= 1

    def test_categorical_plot_creation_with_missing_features(self, sample_processed_data):
        """Test categorical plot creation when some categorical features are missing."""
        # Remove some categorical features
        reduced_data = sample_processed_data.drop(['Employment_Type', 'Zip_Code_Group'], axis=1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock directory paths at all levels
            with patch('sterbling_ai_bias_bounty.config.FIGURES_DIR', temp_path):
                with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                    # Should not crash with missing features
                    generate_eda_plots(reduced_data)
                    
                    # Should still create available plots
                    assert (temp_path / "target_distribution.png").exists()
                    assert (temp_path / "numerical_distributions.png").exists()

    def test_plot_file_output_formats(self, sample_processed_data):
        """Test that plots are saved in correct format and resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bias_output = temp_path / "bias_analysis.png"
            
            with patch('sterbling_ai_bias_bounty.config.PROCESSED_DATA_DIR', temp_path):
                generate_bias_plots(sample_processed_data, bias_output)
                
                # Check that file exists and is PNG format
                assert bias_output.exists()
                assert bias_output.suffix == '.png'
                
                # File should have content (not empty)
                assert bias_output.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__])
                    # Close any figures created
    plt.close('all')

    def test_plot_file_output_formats(self, sample_processed_data):
        """Test that plots are saved in correct format and resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bias_output = temp_path / "bias_analysis.png"
            
            with patch('sterbling_ai_bias_bounty.plots.PROCESSED_DATA_DIR', temp_path):
                generate_bias_plots(sample_processed_data, bias_output)
                
                # Check that file exists and is PNG format
                assert bias_output.exists()
                assert bias_output.suffix == '.png'
                
                # File should have content (not empty)
                assert bias_output.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__])
