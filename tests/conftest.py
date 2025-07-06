import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import json  # Add missing import


# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_loan_data():
    """Create realistic sample loan data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    n_samples = 100
    
    return pd.DataFrame({
        'ID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 80, n_samples),
        'Income': np.random.randint(30000, 150000, n_samples),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Loan_Amount': np.random.randint(50000, 500000, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples),
        'Race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n_samples),
        'Age_Group': np.random.choice(['Under 25', '25-60', 'Over 60'], n_samples),
        'Employment_Type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Citizenship_Status': np.random.choice(['Citizen', 'Permanent Resident', 'Visa Holder'], n_samples),
        'Language_Proficiency': np.random.choice(['Native', 'Fluent', 'Intermediate', 'Basic'], n_samples),
        'Disability_Status': np.random.choice(['None', 'Physical', 'Cognitive', 'Other'], n_samples),
        'Criminal_Record': np.random.choice(['None', 'Minor', 'Major'], n_samples),
        'Zip_Code_Group': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'Loan_Approved': np.random.choice(['Approved', 'Denied'], n_samples)
    })


@pytest.fixture
def data_with_missing_values(sample_loan_data):
    """Create sample data with missing values for testing imputation."""
    df = sample_loan_data.copy()
    
    # Introduce missing values
    missing_indices = np.random.choice(len(df), size=10, replace=False)
    df.loc[missing_indices[:5], 'Age'] = np.nan
    df.loc[missing_indices[5:], 'Loan_Amount'] = np.nan
    
    return df
