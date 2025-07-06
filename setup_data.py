"""
Helper script to set up data directories and show what files are needed.
"""
from pathlib import Path
from sterbling_ai_bias_bounty.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

def setup_directories():
    """Create necessary directories and show what files are needed."""
    
    # Create directories
    directories = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Check for required files
    required_files = [
        RAW_DATA_DIR / "loan_access_dataset.csv",
        RAW_DATA_DIR / "test.csv"
    ]
    
    print("\n📋 Required data files:")
    for file_path in required_files:
        if file_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    if not all(f.exists() for f in required_files):
        print("\n⚠️  Missing data files detected!")
        print("Please add your data files to the data/raw/ directory:")
        print("  1. Copy loan_access_dataset.csv from the notebook/hackathon")
        print("  2. Copy test.csv from the notebook/hackathon")
        print("  3. Place both files in data/raw/ directory")
        print("\nThen run: python loan_model.py")
    else:
        print("\n🎉 All required files found! You can now run:")
        print("  python loan_model.py")

if __name__ == "__main__":
    setup_directories()
