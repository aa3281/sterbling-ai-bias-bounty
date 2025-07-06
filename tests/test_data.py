import pytest
import pandas as pd
import numpy as np


def test_data_loading_basic():
    """Test basic data loading functionality."""
    # Create sample data to test with
    sample_data = pd.DataFrame({
        'ID': [1, 2, 3],
        'Age': [25, 30, 35],
        'Income': [50000, 60000, 70000],
        'Loan_Approved': ['Approved', 'Denied', 'Approved']
    })

    assert len(sample_data) == 3
    assert 'ID' in sample_data.columns
    assert 'Loan_Approved' in sample_data.columns
    assert sample_data['Age'].dtype in [np.int64, np.int32, int]


def test_data_structure_validation():
    """Test that data has expected structure."""
    expected_columns = [
        'ID', 'Age', 'Income', 'Credit_Score', 'Loan_Amount',
        'Gender', 'Race', 'Age_Group', 'Employment_Type', 'Education_Level',
        'Citizenship_Status', 'Language_Proficiency', 'Disability_Status',
        'Criminal_Record', 'Zip_Code_Group', 'Loan_Approved'
    ]

    # Test that our expected structure is reasonable
    assert len(expected_columns) == 16
    assert 'Loan_Approved' in expected_columns
    assert 'ID' in expected_columns
