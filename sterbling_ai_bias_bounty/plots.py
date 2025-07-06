from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typer

from sterbling_ai_bias_bounty.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "analysis_plots.png",
    # -----------------------------------------
):
    """Generate comprehensive visualizations for loan approval analysis."""
    print("Generating plots from loan approval data...")
    
    # Create figures directory if it doesn't exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set default output path if None is provided
    if output_path is None:
        output_path = FIGURES_DIR / "analysis_plots.png"
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Dataset shape: {df.shape}")
    
    # Create figure with subplots (from notebook EDA section)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Loan Approval Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Target distribution (from notebook)
    if 'Loan_Approved' in df.columns:
        approval_counts = df['Loan_Approved'].value_counts()
        axes[0, 0].pie(approval_counts.values, labels=['Denied', 'Approved'], autopct='%1.1f%%')
        axes[0, 0].set_title('Loan Approval Distribution')
    
    # 2. Age distribution (from notebook)
    if 'Age' in df.columns:
        axes[0, 1].hist(df['Age'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Age Distribution')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Frequency')
    
    # 3. Income distribution (from notebook)
    if 'Income' in df.columns:
        axes[0, 2].hist(df['Income'], bins=20, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('Income Distribution')
        axes[0, 2].set_xlabel('Income')
        axes[0, 2].set_ylabel('Frequency')
    
    # 4. Credit Score distribution (from notebook)
    if 'Credit_Score' in df.columns:
        axes[1, 0].hist(df['Credit_Score'], bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('Credit Score Distribution')
        axes[1, 0].set_xlabel('Credit Score')
        axes[1, 0].set_ylabel('Frequency')
    
    # 5. Gender vs Loan Approval (from notebook bias analysis)
    if 'Gender' in df.columns and 'Loan_Approved' in df.columns:
        gender_approval = pd.crosstab(df['Gender'], df['Loan_Approved'])
        gender_approval.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Gender vs Loan Approval')
        axes[1, 1].set_xlabel('Gender')
        axes[1, 1].legend(['Denied', 'Approved'])
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Education vs Loan Approval (from notebook)
    if 'Education_Level' in df.columns and 'Loan_Approved' in df.columns:
        edu_approval = pd.crosstab(df['Education_Level'], df['Loan_Approved'])
        edu_approval.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Education vs Loan Approval')
        axes[1, 2].set_xlabel('Education Level')
        axes[1, 2].legend(['Denied', 'Approved'])
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    # 7. Employment vs Loan Approval (from notebook)
    if 'Employment_Type' in df.columns and 'Loan_Approved' in df.columns:
        emp_approval = pd.crosstab(df['Employment_Type'], df['Loan_Approved'])
        emp_approval.plot(kind='bar', ax=axes[2, 0])
        axes[2, 0].set_title('Employment vs Loan Approval')
        axes[2, 0].set_xlabel('Employment Type')
        axes[2, 0].legend(['Denied', 'Approved'])
        axes[2, 0].tick_params(axis='x', rotation=45)
    
    # 8. Loan Amount vs Income scatter (from notebook)
    if 'Loan_Amount' in df.columns and 'Income' in df.columns:
        scatter = axes[2, 1].scatter(df['Income'], df['Loan_Amount'], 
                                   c=df['Loan_Approved'] if 'Loan_Approved' in df.columns else 'blue',
                                   alpha=0.6, cmap='RdYlBu')
        axes[2, 1].set_title('Loan Amount vs Income')
        axes[2, 1].set_xlabel('Income')
        axes[2, 1].set_ylabel('Loan Amount')
        if 'Loan_Approved' in df.columns:
            plt.colorbar(scatter, ax=axes[2, 1])
    
    # 9. Correlation heatmap (from notebook)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[2, 2], cbar_kws={'shrink': 0.8})
        axes[2, 2].set_title('Feature Correlation Matrix')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to: {output_path}")
    
    # Generate bias analysis plots (from notebook bias section)
    bias_output_path = FIGURES_DIR / "bias_analysis.png"
    generate_bias_plots(df, bias_output_path)
    
    # Generate model interpretability plots (from notebook)
    generate_model_interpretability_plots(df)
    
    # Generate EDA plots exactly from notebook Cell 3
    generate_eda_plots(df)
    
    # Generate model evaluation plots (now implemented)
    generate_model_evaluation_plots()
    
    print("Plot generation complete.")


def generate_bias_plots(df, output_path):
    """Generate specific bias analysis visualizations from notebook."""
    print("Generating bias analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('AI Bias Analysis - Loan Approval', fontsize=16, fontweight='bold')
    
    # Protected attributes for bias analysis (from notebook)
    protected_attrs = ['Gender', 'Race', 'Age_Group', 'Citizenship_Status']
    
    for i, attr in enumerate(protected_attrs):
        if attr in df.columns and 'Loan_Approved' in df.columns:
            row, col = i // 2, i % 2
            
            # Calculate approval rates by protected attribute (from notebook)
            approval_rates = df.groupby(attr)['Loan_Approved'].mean()
            
            approval_rates.plot(kind='bar', ax=axes[row, col])
            axes[row, col].set_title(f'Approval Rate by {attr}')
            axes[row, col].set_ylabel('Approval Rate')
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add horizontal line for overall approval rate (from notebook)
            overall_rate = df['Loan_Approved'].mean()
            axes[row, col].axhline(y=overall_rate, color='red', linestyle='--', 
                                 label=f'Overall Rate: {overall_rate:.2%}')
            axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bias analysis plots saved to: {output_path}")


def generate_model_interpretability_plots(df):
    """Generate SHAP, LIME, and feature importance plots using our superior feature engineering."""
    print("Generating model interpretability plots...")
    
    # Check if trained model exists
    from sterbling_ai_bias_bounty.config import MODELS_DIR
    model_path = MODELS_DIR / "model.pkl"
    
    if not model_path.exists():
        print("No trained model found. Skipping interpretability plots.")
        return
    
    try:
        import joblib
        model = joblib.load(model_path)
        
        # Use our superior 27-feature engineered dataset
        features_path = PROCESSED_DATA_DIR / "features.csv"
        if not features_path.exists():
            print("Features file not found. Skipping interpretability plots.")
            return
        
        # Load our engineered features (27 features - better than notebook's 14)
        X_train = pd.read_csv(features_path)
        
        # Create test data with same structure 
        X_test = X_train.sample(min(1000, len(X_train)), random_state=42)  # Reasonable sample size
        
        feature_cols = X_train.columns.tolist()
        
        print(f"Using {len(feature_cols)} engineered features (superior to notebook's basic approach)")
        
        # 1. Feature Importance Plots - enhanced version
        generate_feature_importance_plots(model, feature_cols)
        
        # 2. SHAP Analysis - with our better features
        generate_shap_plots(model, X_train, X_test, feature_cols)
        
        # 3. LIME Analysis - with our better features
        generate_lime_plots(model, X_train, X_test, feature_cols)
        
    except Exception as e:
        print(f"Error generating interpretability plots: {e}")
        import traceback
        traceback.print_exc()


def generate_eda_plots(df):
    """Generate EDA plots exactly matching notebook Cell 3."""
    print("Generating EDA plots from notebook...")
    
    # Target distribution (from notebook)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Loan_Approved', data=df)
    plt.title('Loan Approval Status Distribution')
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Numerical feature distribution (from notebook)
    plt.figure(figsize=(12, 8))
    df.hist(bins=10, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numerical_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Categorical feature relationships (exactly from notebook)
    categorical_features = ['Education_Level', 'Gender', 'Employment_Type', 'Citizenship_Status', 'Zip_Code_Group']
    
    for feature in categorical_features:
        if feature in df.columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=feature, hue='Loan_Approved', data=df)
            plt.title(f'{feature} vs Loan Status')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{feature.lower()}_vs_loan.png", dpi=300, bbox_inches='tight')
            plt.close()


def generate_model_evaluation_plots():
    """Generate proper model evaluation plots with validation data."""
    print("Generating model evaluation plots...")
    
    # Check if we have models and validation data
    from sterbling_ai_bias_bounty.config import MODELS_DIR
    all_models_path = MODELS_DIR / "all_models.pkl"
    
    if not all_models_path.exists():
        print("No trained models found. Skipping model evaluation plots.")
        return
    
    try:
        import joblib
        from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
        
        # Load all models and validation data
        all_data = joblib.load(all_models_path)
        models = all_data['models']
        X_val, y_val = all_data['validation_data']
        scaler = all_data['scaler']
        
        # Generate evaluation for each model
        model_results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            requires_scaling = model_info['requires_scaling']
            
            # Prepare validation data
            if requires_scaling:
                X_val_processed = scaler.transform(X_val)
            else:
                X_val_processed = X_val
            
            # Generate predictions
            y_pred = model.predict(X_val_processed)
            y_prob = model.predict_proba(X_val_processed)[:, 1]
            
            model_results[model_name] = {
                'y_pred': y_pred,
                'y_prob': y_prob,
                'auc': model_info['auc']
            }
        
        # 1. Confusion Matrix for best model
        best_model = max(model_results.keys(), key=lambda k: model_results[k]['auc'])
        plt.figure(figsize=(8, 6))
        conf_mat = confusion_matrix(y_val, model_results[best_model]['y_pred'])
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{best_model} Confusion Matrix (Validation Set)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {FIGURES_DIR / 'confusion_matrix.png'}")
        
        # 2. ROC Curve Comparison (proper approach)
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            fpr, tpr, _ = roc_curve(y_val, results['y_prob'])
            auc_score = results['auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0,1],[0,1],'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison (Validation Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(FIGURES_DIR / "roc_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC comparison saved to: {FIGURES_DIR / 'roc_comparison.png'}")
        
        # 3. Model Performance Summary
        create_proper_model_summary_table(model_results, y_val)
        
    except Exception as e:
        print(f"Error generating model evaluation plots: {e}")
        import traceback
        traceback.print_exc()


def create_proper_model_summary_table(model_results, y_val):
    """Create a proper model performance summary table."""
    print("Creating model summary table...")
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Create comprehensive summary DataFrame
        summary_data = []
        for model_name, results in model_results.items():
            y_pred = results['y_pred']
            auc_score = results['auc']
            
            summary_data.append({
                'Model': model_name,
                'AUC': f"{auc_score:.4f}",
                'Accuracy': f"{accuracy_score(y_val, y_pred):.4f}",
                'Precision': f"{precision_score(y_val, y_pred):.4f}",
                'Recall': f"{recall_score(y_val, y_pred):.4f}",
                'F1-Score': f"{f1_score(y_val, y_pred):.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.title('Model Performance Summary (Validation Set)', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(FIGURES_DIR / "model_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model summary table saved to: {FIGURES_DIR / 'model_summary_table.png'}")
        
    except Exception as e:
        print(f"Error creating model summary table: {e}")


def generate_all_feature_importance_plots(model, feature_names):
    """Generate feature importance plots for all models from notebook Cell 22-24."""
    print("Generating comprehensive feature importance plots...")
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # 1. Seaborn horizontal barplot (main style from notebook)
            plt.figure(figsize=(10, 8))
            sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
            plt.title('Feature Importance (Detailed View)')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "feature_importance_detailed.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Top 10 features only (cleaner view)
            plt.figure(figsize=(8, 6))
            top_10_indices = indices[:10]
            sns.barplot(x=importances[top_10_indices], y=[feature_names[i] for i in top_10_indices])
            plt.title('Top 10 Most Important Features')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "feature_importance_top10.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Feature importance plots saved to: {FIGURES_DIR}")
            
            # 3. XGBoost built-in plot style (if XGBoost model)
            model_name = type(model).__name__
            if 'XGB' in model_name:
                try:
                    from xgboost import plot_importance
                    plt.figure(figsize=(8, 10))
                    plot_importance(model, height=0.6)
                    plt.title('XGBoost Feature Importance (Built-in)')
                    plt.tight_layout()
                    plt.savefig(FIGURES_DIR / "xgboost_builtin_importance.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"XGBoost built-in importance saved to: {FIGURES_DIR / 'xgboost_builtin_importance.png'}")
                except Exception as e:
                    print(f"Could not create XGBoost built-in plot: {e}")
        
    except Exception as e:
        print(f"Error generating feature importance plots: {e}")


# Update the feature importance function call
def generate_feature_importance_plots(model, feature_names):
    """Generate feature importance visualizations exactly from notebook Cell 22."""
    # Call the comprehensive version
    generate_all_feature_importance_plots(model, feature_names)

def generate_shap_plots(model, X_train, X_test, feature_names):
    """Generate SHAP visualizations exactly from notebook Cells 25-29."""
    print("Generating SHAP plots...")
    
    try:
        import shap
        
        # Match the notebook's exact approach - Cell 29 (Random Forest example)
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        
        # Summary plot exactly like notebook
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        shap_summary_path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to: {shap_summary_path}")
        
        # Force plot exactly like notebook Cell 30
        # The notebook shows this is an interactive plot, so we'll save as HTML
        try:
            shap_html = shap.plots.force(shap_values[0], matplotlib=False)
            
            # Save as HTML (like notebook displays)
            force_html_path = FIGURES_DIR / "shap_force_plot.html"
            shap.save_html(force_html_path, shap_html)
            print(f"SHAP force plot (HTML) saved to: {force_html_path}")
        except:
            print("SHAP HTML force plot failed")
        
        # Also try matplotlib version
        try:
            plt.figure(figsize=(20, 3))
            shap.plots.force(shap_values[0], matplotlib=True, show=False)
            force_plot_path = FIGURES_DIR / "shap_force_plot.png"
            plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP force plot (PNG) saved to: {force_plot_path}")
        except:
            print("SHAP matplotlib force plot failed, but may have HTML version")
        
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
    except Exception as e:
        print(f"Error generating SHAP plots: {e}")


def generate_lime_plots(model, X_train, X_test, feature_names):
    """Generate LIME visualizations exactly from notebook Cells 32-34."""
    print("Generating LIME plots...")
    
    try:
        from lime.lime_tabular import LimeTabularExplainer
        
        # Initialize LIME explainer exactly like notebook Cell 32
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns.tolist(),  # Use exact column names from training
            class_names=['Not Approved', 'Approved'],  # Match notebook exactly
            mode='classification'
        )
        
        # Explain first instance exactly like notebook Cell 33
        i = 0  # Match notebook variable name
        test_instance = X_test.iloc[i].values  # Get first test instance
        
        # Use model.predict_proba directly like notebook
        exp = explainer.explain_instance(
            data_row=test_instance,
            predict_fn=model.predict_proba,
            num_features=len(feature_names)  # Show all features like notebook
        )
        
        # Save LIME explanation exactly like notebook Cell 34
        fig = exp.as_pyplot_figure()
        lime_path = FIGURES_DIR / "lime_explanation.png"
        fig.savefig(lime_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"LIME explanation saved to: {lime_path}")
        
    except ImportError:
        print("LIME not available. Install with: pip install lime")
    except Exception as e:
        print(f"Error generating LIME plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    app()
    app()
if __name__ == "__main__":
    app()
    app()
