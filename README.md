# Sterbling AI Bias Bounty

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A classification model to detect and explain unusual patterns in AI decision-making for mortgage loan approvals

## Demo and Screenshot
![](reports/figures/analysis_plots.png)

## Key Features
- Comprehensive data preprocessing with bias-aware feature engineering
- Multiple machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Automated hyperparameter tuning with cross-validation
- Bias detection and analysis across protected attributes
- Interactive visualizations and explainability with SHAP and LIME
- Modular pipeline architecture for easy experimentation

## Usage

### Run the Pipeline

```bash
# FIRST: Set up directories and check data files
python setup_data.py

# THEN: Run the full pipeline (default) - executes all 5 steps automatically
python loan_model.py

# Or run specific steps individually (after data is available)
python loan_model.py data
python loan_model.py feature-engineering
python loan_model.py train-model
python loan_model.py predict-model
python loan_model.py visualize

# See all available commands
python loan_model.py --help
```

### Alternative: Use Make commands

```bash
# Run individual steps
make data
make features
make train
make predict
make plots

# Run full pipeline
make pipeline
```

## Development
Easily set up a local development environment!

1. Clone the repo
    ```bash
    git clone git@github.com:aa3281/sterbling-ai-bias-bounty.git
    ```
2. Install the Python packages 
    ```bash
    pip install -e .
    ```
3. Add your raw data files to `data/raw/`
    ```bash
    # Place your data files:
    # data/raw/loan_access_dataset.csv
    # data/raw/test.csv
    ```

## Built With
- Python 3.10
- Pandas & NumPy for data manipulation
- Scikit-learn for machine learning
- XGBoost & LightGBM for gradient boosting
- SHAP & LIME for model explainability
- Matplotlib & Seaborn for visualization
- Typer for CLI interface
- Loguru for logging

## Troubleshooting

### Memory Issues During Training
If you encounter `TerminatedWorkerError` or memory issues:

1. **Reduce dataset size for testing:**
   ```bash
   # Use a smaller sample of your data first
   python loan_model.py train-model
   ```

2. **Monitor memory usage:**
   ```bash
   # Check available memory
   free -h
   # Monitor during training
   top -p $(pgrep -f python)
   ```

3. **Reduce parallelization:**
   The training script automatically uses `n_jobs=1` to prevent worker crashes.

### Common Issues
- **FileNotFoundError**: Make sure raw data files exist in `data/raw/`
- **Missing dependencies**: Run `pip install -e .` to install all requirements
- **Feature mismatch**: Ensure test data goes through same preprocessing as training data
- **Pipeline runs too fast**: Check if raw data files exist in `data/raw/` - without them, steps may be skipped
- **No outputs generated**: Verify all directories exist: `data/processed/`, `models/`, `reports/figures/`

### Verifying Pipeline Execution

To ensure the pipeline is working correctly:

```bash
# Check if all output files are generated
ls -la data/processed/    # Should contain dataset.csv, features.csv, labels.csv
ls -la models/           # Should contain model.pkl, scaler.pkl, model_metadata.json
ls -la reports/figures/  # Should contain analysis_plots.png, shap_summary.png, etc.

# Monitor pipeline execution with verbose logging
python loan_model.py --help  # See available commands
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         sterbling_ai_bias_bounty and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── loan_model.py      <- Main pipeline orchestration script
│
└── sterbling_ai_bias_bounty   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sterbling_ai_bias_bounty a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## To-Do
- [ ] Add support for additional bias metrics (equalized odds, demographic parity)
- [ ] Implement automated bias mitigation techniques
- [ ] Add model interpretability dashboard
- [ ] Create comprehensive documentation with examples
- [ ] Add unit tests and integration tests
- [ ] Implement model versioning and experiment tracking

## Feedback and Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. All contributions are welcome!

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork it 
2. Create your feature branch (`git checkout -b feature/amazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazingFeature`)
5. Create a new Pull Request

## Acknowledgements
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for project template
- [SHAP](https://shap.readthedocs.io/) for model explainability
- [LIME](https://lime-ml.readthedocs.io/) for local interpretability
- Sterbling AI Bias Bounty Challenge organizers

--------

