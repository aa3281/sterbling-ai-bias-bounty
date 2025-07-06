# Sterbling AI Bias Bounty

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A production-ready classification pipeline to detect and explain AI bias patterns in mortgage loan approval decisions, with comprehensive interpretability tools and statistical bias analysis.

## Demo and Screenshot

### Pipeline Execution Demo
![Pipeline Demo](reports/figures/pipeline_demo.gif)
*Complete bias detection pipeline showing data processing, model training, and comprehensive bias analysis visualization generation.*

### Comprehensive Analysis Dashboard
![Analysis Dashboard](reports/figures/analysis_plots.png)
*Main analysis dashboard showing EDA, feature distributions, and bias patterns across demographic groups with proper categorical labels.*

### Bias Detection Results
![Bias Detection Results](reports/figures/bias_analysis.png)
*Statistical bias analysis showing approval rate disparities across protected attributes with sample sizes and significance indicators.*

## What We Built - Comprehensive Bias Detection System

We developed a complete AI bias detection pipeline that transforms a basic Jupyter notebook into a production-ready system for identifying and quantifying bias in loan approval decisions:

### Advanced AI Bias Detection Framework
- **Statistical Bias Analysis**: Quantitative measurement of approval rate disparities across demographic groups with proper encoding mappings
- **Demographic Pattern Recognition**: Automated identification of systematic disparities in loan approval rates across protected attributes
- **False Positive/Negative Analysis**: Deep analysis of prediction errors and their differential impact on demographic groups  
- **Statistical Significance Testing**: Proper hypothesis testing with sample size considerations and confidence intervals
- **Interactive Visualizations**: Dynamic bias analysis plots with meaningful category labels and statistical context

## How the Model Works - Production Architecture

### Interpretable Multi-Model Pipeline
Our production system implements multiple algorithms with comprehensive comparison and bias analysis:

**Model Ensemble:**
- **Random Forest**: Robust feature importance with bias-aware analysis
- **Logistic Regression**: Transparent coefficient interpretation with proper scaling
- **XGBoost**: Advanced gradient boosting with built-in feature importance visualization
- **LightGBM**: Efficient gradient boosting optimized for bias detection workflows

### Enhanced Feature Engineering & Data Processing
- **Intelligent Missing Value Handling**: Mode imputation for categorical, median for numerical
- **Bias-Aware Encoding**: Preserves original categorical mappings for meaningful bias analysis
- **Statistical Validation**: Proper train/validation splits with stratified sampling
- **Feature Lineage Tracking**: Complete audit trail of data transformations for bias investigation

### Model Selection & Evaluation Criteria
- **Multi-Metric Evaluation**: AUC, accuracy, precision, recall, F1-score comparison
- **Cross-Validation**: Robust 3-fold CV with bias-aware sampling
- **Demographic Parity Analysis**: Performance evaluation across protected groups
- **Interpretability Requirements**: All models provide feature importance and explanation capabilities

## How We Approached Fairness - Advanced Bias Detection

### Production Red Team Methodology
We implemented a systematic bias detection approach with:
- **Collaborative Code Review**: Cross-validation of bias detection logic and statistical assumptions
- **Automated Bias Scanning**: Pipeline-integrated bias detection across multiple demographic dimensions
- **Statistical Rigor**: Proper hypothesis testing with multiple comparison corrections

### Comprehensive AI Bias Tools Integration

**Data-Level Analysis:**
- Demographic distribution analysis with encoding preservation
- Historical bias pattern identification in approval rates
- Statistical parity testing across protected attributes

**Model-Level Interpretability:**
- SHAP (SHapley Additive exPlanations) with corrected v0.20+ API usage
- LIME (Local Interpretable Model-agnostic Explanations) with proper feature name handling
- Multi-model feature importance comparison and bias impact analysis

**Visualization-Level Insights:**
- Comprehensive bias analysis plots with meaningful category labels
- Statistical significance indicators and sample size annotations
- ROC curve analysis across demographic subgroups

### Production Pipeline Features
- **Automated Encoding Mapping**: Preserves categorical value mappings for bias analysis
- **Error Handling**: Graceful handling of missing dependencies and data validation
- **Modular Architecture**: Individual step execution with comprehensive logging
- **Reproducible Results**: Fixed random seeds and version-controlled data processing

## What Biases Were Discovered - Statistical Evidence

### Key Statistical Findings
Our comprehensive analysis revealed significant demographic patterns:

![SHAP Summary Analysis](reports/figures/shap_summary.png)
*SHAP analysis revealing feature importance patterns across demographic and creditworthiness variables.*

### Feature Importance Analysis Across Models

Understanding which features drive loan approval decisions is critical for bias detection:

![Random Forest Feature Importance](reports/figures/feature_importance_detailed.png)
*Random Forest feature importance showing the relative contribution of each feature to loan approval decisions. Notice how demographic features rank compared to financial metrics.*

![XGBoost Built-in Feature Importance](reports/figures/xgboost_builtin_importance.png)
*XGBoost built-in feature importance plot demonstrating which features the gradient boosting algorithm considers most predictive. The prominence of certain attributes helps identify potential bias sources.*

![Top 10 Feature Importance](reports/figures/feature_importance_top10.png)
*Top 10 most important features across models, providing a clear view of the most influential decision factors in loan approval predictions.*

### Global Model Interpretability
![SHAP Summary Analysis](reports/figures/shap_summary.png)
*SHAP summary plot revealing feature importance and impact on loan approval decisions across the entire dataset. Each dot represents a prediction, showing both feature importance and directional impact.*

### Individual Prediction Explanations
![LIME Explanation Example](reports/figures/lime_explanation.png)
*LIME explanation for a specific loan application showing how individual features contributed to the decision. Red bars indicate features that decrease approval likelihood, while green bars show features that increase it.*

## Model Performance Analysis

### Multi-Model ROC Curve Comparison
![ROC Curve Comparison](reports/figures/roc_comparison.png)
*ROC curve comparison across all models (Random Forest, XGBoost, LightGBM, Logistic Regression) showing performance differences and potential bias amplification patterns across algorithms.*

### Model Performance Metrics
![Model Performance Summary](reports/figures/model_summary_table.png)
*Comprehensive performance summary showing accuracy, precision, recall, F1-score, and AUC metrics. This table helps identify trade-offs between predictive performance and potential bias amplification across different algorithms.*

### Prediction Accuracy Analysis
![Confusion Matrix](reports/figures/confusion_matrix.png)
*Confusion matrix for the best-performing model showing prediction accuracy across different classes. The matrix helps identify systematic prediction errors that may disproportionately affect certain demographic groups.*

## Key Features
- ✅ **Production-Ready Pipeline**: Complete MLOps workflow with individual step execution
- ✅ **Advanced Bias Detection**: Statistical analysis across multiple protected attributes  
- ✅ **Multi-Model Comparison**: Random Forest, XGBoost, LightGBM, Logistic Regression
- ✅ **Comprehensive Interpretability**: SHAP, LIME, and feature importance analysis
- ✅ **Proper Data Handling**: Encoding preservation, missing value handling, validation splits
- ✅ **Statistical Rigor**: Hypothesis testing, confidence intervals, sample size analysis
- ✅ **Error Handling**: Graceful degradation and comprehensive logging
- ✅ **Visualization Suite**: 15+ bias analysis and model evaluation plots

## Usage

### Quick Start
```bash
# Complete pipeline execution (recommended)
python loan_model.py

# Individual step execution for debugging
python loan_model.py data
python loan_model.py feature-engineering  
python loan_model.py train-model
python loan_model.py predict-model
python loan_model.py visualize
```

### Expected Outputs
The pipeline generates comprehensive analysis artifacts:
```bash
data/processed/          # Clean datasets with encoding mappings
├── dataset.csv          # Processed training data
├── features.csv         # Engineered features  
├── labels.csv          # Target variables
├── encoding_mappings.json  # Categorical mappings for bias analysis
└── test_processed.csv   # Processed test data

models/                  # Trained models and metadata
├── model.pkl           # Best performing model
├── all_models.pkl      # All trained models with validation data
├── scaler.pkl          # Feature scaling parameters
└── model_metadata.json # Model selection and performance data

reports/figures/         # Comprehensive visualization suite
├── analysis_plots.png  # Main EDA dashboard
├── bias_analysis.png   # Demographic bias analysis
├── confusion_matrix.png # Model prediction accuracy
├── roc_comparison.png  # Multi-model performance
├── shap_summary.png    # Global feature importance
├── lime_explanation.png # Individual prediction explanation
└── feature_importance_*.png # Model-specific importance plots
```

## Development
Production-ready development environment setup:

1. **Repository Setup**
    ```bash
    git clone git@github.com:aa3281/sterbling-ai-bias-bounty.git
    cd sterbling-ai-bias-bounty
    ```

2. **Environment Configuration**
    ```bash
    # Python 3.10+ required
    pip install -e .
    ```

3. **Data Requirements**
    ```bash
    # Required data files:
    data/raw/loan_access_dataset.csv  # Training data
    data/raw/test.csv                 # Test data
    ```

## Built With
- **Core ML**: scikit-learn, XGBoost, LightGBM, pandas, NumPy
- **Bias Detection**: SHAP v0.20+, LIME, statistical analysis tools
- **Visualization**: Matplotlib, Seaborn, comprehensive plotting pipeline
- **Infrastructure**: Typer CLI, Loguru logging, automated error handling
- **Development**: Python 3.10+, JSON-based configuration, modular architecture

## Troubleshooting

### Common Issues & Solutions
- **Memory Issues**: Pipeline automatically uses `n_jobs=1` and optimized sampling
- **Missing Data Files**: Clear error messages with specific file path requirements
- **Dependency Conflicts**: Graceful degradation with installation guidance
- **Visualization Errors**: Comprehensive error handling with fallback options
- **Model Training Failures**: Detailed logging for debugging and recovery

### Performance Monitoring
```bash
# Validate pipeline outputs
ls -la data/processed/    # Verify data processing
ls -la models/           # Check model artifacts  
ls -la reports/figures/  # Confirm visualization generation
```

## Impact and Future Work

### Bias Mitigation Insights
Our analysis provides actionable insights for fair lending practices:
- **Demographic Monitoring**: Automated tracking of approval rate disparities
- **Model Comparison**: Identification of algorithms that amplify or reduce bias
- **Statistical Evidence**: Rigorous quantification of bias patterns for regulatory compliance
- **Interpretability**: Clear explanations of individual loan decisions for transparency

### Research Contributions
- **Production Bias Pipeline**: Complete MLOps workflow for bias detection
- **Statistical Framework**: Rigorous approach to bias measurement and significance testing
- **Educational Resource**: Transformation of educational notebook into production system
- **Open Source**: Fully documented, reproducible bias detection methodology

## Acknowledgements
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for project template
- [SHAP](https://shap.readthedocs.io/) for explainable AI framework
- [LIME](https://lime-ml.readthedocs.io/) for local interpretability
- Sterbling AI Bias Bounty Challenge organizers
- **Team**: Alessandra Adina & Sydney Nicole Calo

--------
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

