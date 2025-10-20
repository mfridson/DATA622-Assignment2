"""
Machine Learning Experimentation Assignment
Comparing Decision Trees, Random Forest, and AdaBoost
Author: Marc Fridson
Date: October 19, 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset (using breast cancer dataset as example - replace with your dataset)
print("Loading and preparing data...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Initial train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dataset shape: {X.shape}")
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}\n")

# ==============================================================================
# EXPERIMENT TRACKING SYSTEM
# ==============================================================================

class ExperimentTracker:
    """System to track and document all experiments"""
    
    def __init__(self):
        self.experiments = []
    
    def log_experiment(self, name, objective, variation, metric, model, results):
        """Log experiment details and results"""
        experiment = {
            'Name': name,
            'Objective': objective,
            'Variation': variation,
            'Evaluation_Metric': metric,
            'Model': model.__class__.__name__,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1'],
            'AUC_ROC': results['auc_roc'],
            'Cross_Val_Mean': results['cv_mean'],
            'Cross_Val_Std': results['cv_std'],
            'Train_Score': results['train_score'],
            'Test_Score': results['test_score']
        }
        self.experiments.append(experiment)
        print(f"✓ Experiment logged: {name}")
    
    def get_results_table(self):
        """Return experiments as DataFrame"""
        return pd.DataFrame(self.experiments)

# Initialize tracker
tracker = ExperimentTracker()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0,
        'train_score': model.score(X_train, y_train),
        'test_score': model.score(X_test, y_test)
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    results['cv_mean'] = cv_scores.mean()
    results['cv_std'] = cv_scores.std()
    
    return results

# ==============================================================================
# DECISION TREE EXPERIMENTS
# ==============================================================================

print("="*80)
print("DECISION TREE EXPERIMENTS")
print("="*80)

# Experiment DT-1: Baseline Decision Tree
print("\n[DT-1] Running baseline Decision Tree...")
objective_dt1 = "Establish baseline performance with default Decision Tree parameters"
variation_dt1 = "Default parameters (no max_depth limit, min_samples_split=2)"

dt_baseline = DecisionTreeClassifier(random_state=42)
results_dt1 = evaluate_model(dt_baseline, X_train, X_test, y_train, y_test)

tracker.log_experiment(
    name="DT-1: Baseline",
    objective=objective_dt1,
    variation=variation_dt1,
    metric="Accuracy, F1-Score",
    model=dt_baseline,
    results=results_dt1
)

print(f"Accuracy: {results_dt1['accuracy']:.4f}")
print(f"F1-Score: {results_dt1['f1']:.4f}")
print(f"Bias (Training Score): {results_dt1['train_score']:.4f}")
print(f"Variance (Train-Test Gap): {results_dt1['train_score'] - results_dt1['test_score']:.4f}")

# Experiment DT-2: Pruned Decision Tree with Feature Selection
print("\n[DT-2] Running pruned Decision Tree with feature selection...")
objective_dt2 = "Reduce overfitting through pruning and feature selection to improve generalization"
variation_dt2 = "max_depth=5, min_samples_split=20, top 15 features selected"

# Feature selection
selector = SelectKBest(f_classif, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

dt_pruned = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
results_dt2 = evaluate_model(dt_pruned, X_train_selected, X_test_selected, y_train, y_test)

tracker.log_experiment(
    name="DT-2: Pruned + Feature Selection",
    objective=objective_dt2,
    variation=variation_dt2,
    metric="Accuracy, F1-Score",
    model=dt_pruned,
    results=results_dt2
)

print(f"Accuracy: {results_dt2['accuracy']:.4f}")
print(f"F1-Score: {results_dt2['f1']:.4f}")
print(f"Bias (Training Score): {results_dt2['train_score']:.4f}")
print(f"Variance (Train-Test Gap): {results_dt2['train_score'] - results_dt2['test_score']:.4f}")

# ==============================================================================
# RANDOM FOREST EXPERIMENTS
# ==============================================================================

print("\n" + "="*80)
print("RANDOM FOREST EXPERIMENTS")
print("="*80)

# Experiment RF-1: Small Forest with Normalized Data
print("\n[RF-1] Running Random Forest with normalized data...")
objective_rf1 = "Test impact of data normalization on ensemble performance with small forest"
variation_rf1 = "n_estimators=50, normalized features using MinMaxScaler"

# Normalize data
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

rf_small = RandomForestClassifier(n_estimators=50, random_state=42)
results_rf1 = evaluate_model(rf_small, X_train_norm, X_test_norm, y_train, y_test)

tracker.log_experiment(
    name="RF-1: Small Forest + Normalization",
    objective=objective_rf1,
    variation=variation_rf1,
    metric="Accuracy, AUC-ROC",
    model=rf_small,
    results=results_rf1
)

print(f"Accuracy: {results_rf1['accuracy']:.4f}")
print(f"AUC-ROC: {results_rf1['auc_roc']:.4f}")
print(f"Bias (Training Score): {results_rf1['train_score']:.4f}")
print(f"Variance (Train-Test Gap): {results_rf1['train_score'] - results_rf1['test_score']:.4f}")

# Experiment RF-2: Optimized Random Forest with Grid Search
print("\n[RF-2] Running optimized Random Forest with hyperparameter tuning...")
objective_rf2 = "Find optimal hyperparameters to maximize model performance"
variation_rf2 = "Grid search: n_estimators=[100,200], max_depth=[10,20,None], min_samples_split=[2,5]"

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
results_rf2 = evaluate_model(best_rf, X_train, X_test, y_train, y_test)

tracker.log_experiment(
    name="RF-2: Grid Search Optimized",
    objective=objective_rf2,
    variation=variation_rf2,
    metric="F1-Score (optimization target)",
    model=best_rf,
    results=results_rf2
)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy: {results_rf2['accuracy']:.4f}")
print(f"F1-Score: {results_rf2['f1']:.4f}")
print(f"Bias (Training Score): {results_rf2['train_score']:.4f}")
print(f"Variance (Train-Test Gap): {results_rf2['train_score'] - results_rf2['test_score']:.4f}")

# ==============================================================================
# ADABOOST EXPERIMENTS
# ==============================================================================

print("\n" + "="*80)
print("ADABOOST EXPERIMENTS")
print("="*80)

# Experiment AB-1: AdaBoost with Weak Learners
print("\n[AB-1] Running AdaBoost with weak learners...")
objective_ab1 = "Test boosting with intentionally weak base estimators"
variation_ab1 = "n_estimators=30, max_depth=1 for base estimator (decision stumps)"

base_weak = DecisionTreeClassifier(max_depth=1)
ada_weak = AdaBoostClassifier(estimator=base_weak, n_estimators=30, random_state=42)
results_ab1 = evaluate_model(ada_weak, X_train, X_test, y_train, y_test)

tracker.log_experiment(
    name="AB-1: Weak Learners",
    objective=objective_ab1,
    variation=variation_ab1,
    metric="Accuracy, Precision",
    model=ada_weak,
    results=results_ab1
)

print(f"Accuracy: {results_ab1['accuracy']:.4f}")
print(f"Precision: {results_ab1['precision']:.4f}")
print(f"Bias (Training Score): {results_ab1['train_score']:.4f}")
print(f"Variance (Train-Test Gap): {results_ab1['train_score'] - results_ab1['test_score']:.4f}")

# Experiment AB-2: AdaBoost with Increased Complexity and Different Split
print("\n[AB-2] Running AdaBoost with increased complexity and different train-test split...")
objective_ab2 = "Increase model complexity and test generalization with different data split"
variation_ab2 = "n_estimators=100, learning_rate=0.5, 70-30 train-test split"

# Different train-test split
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ada_complex = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
results_ab2 = evaluate_model(ada_complex, X_train_70, X_test_30, y_train_70, y_test_30)

tracker.log_experiment(
    name="AB-2: Complex + Different Split",
    objective=objective_ab2,
    variation=variation_ab2,
    metric="Recall, AUC-ROC",
    model=ada_complex,
    results=results_ab2
)

print(f"Accuracy: {results_ab2['accuracy']:.4f}")
print(f"Recall: {results_ab2['recall']:.4f}")
print(f"Bias (Training Score): {results_ab2['train_score']:.4f}")
print(f"Variance (Train-Test Gap): {results_ab2['train_score'] - results_ab2['test_score']:.4f}")

# ==============================================================================
# RESULTS SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)

# Get results table
results_df = tracker.get_results_table()

# Display key metrics
summary_cols = ['Name', 'Accuracy', 'F1_Score', 'Cross_Val_Mean', 'Train_Score', 'Test_Score']
print("\n", results_df[summary_cols].to_string(index=False))

# Find optimal model
best_model_idx = results_df['F1_Score'].idxmax()
best_model = results_df.loc[best_model_idx, 'Name']
best_f1 = results_df.loc[best_model_idx, 'F1_Score']

print(f"\n✅ OPTIMAL MODEL: {best_model}")
print(f"   Best F1-Score: {best_f1:.4f}")

# Bias-Variance Analysis
print("\n" + "="*80)
print("BIAS-VARIANCE ANALYSIS")
print("="*80)

for idx, row in results_df.iterrows():
    bias = row['Train_Score']
    variance = row['Train_Score'] - row['Test_Score']
    print(f"\n{row['Name']}:")
    print(f"  Bias (Training Score): {bias:.4f}")
    print(f"  Variance (Overfit Gap): {variance:.4f}")
    if variance > 0.05:
        print(f"  ⚠️  High variance detected - model is overfitting")
    elif bias < 0.85:
        print(f"  ⚠️  High bias detected - model is underfitting")
    else:
        print(f"  ✓  Good bias-variance balance")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Accuracy comparison
ax1 = axes[0, 0]
ax1.bar(range(len(results_df)), results_df['Accuracy'])
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels(results_df['Name'], rotation=45, ha='right')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.axhline(y=results_df['Accuracy'].mean(), color='r', linestyle='--', label='Mean')
ax1.legend()

# Plot 2: F1-Score comparison
ax2 = axes[0, 1]
ax2.bar(range(len(results_df)), results_df['F1_Score'], color='orange')
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels(results_df['Name'], rotation=45, ha='right')
ax2.set_ylabel('F1-Score')
ax2.set_title('Model F1-Score Comparison')
ax2.axhline(y=results_df['F1_Score'].mean(), color='r', linestyle='--', label='Mean')
ax2.legend()

# Plot 3: Train vs Test Score (Bias-Variance)
ax3 = axes[1, 0]
x_pos = np.arange(len(results_df))
width = 0.35
ax3.bar(x_pos - width/2, results_df['Train_Score'], width, label='Train', color='green')
ax3.bar(x_pos + width/2, results_df['Test_Score'], width, label='Test', color='blue')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['Name'], rotation=45, ha='right')
ax3.set_ylabel('Score')
ax3.set_title('Train vs Test Score (Bias-Variance Trade-off)')
ax3.legend()

# Plot 4: Cross-validation scores with error bars
ax4 = axes[1, 1]
ax4.errorbar(range(len(results_df)), results_df['Cross_Val_Mean'], 
             yerr=results_df['Cross_Val_Std'], fmt='o-', capsize=5)
ax4.set_xticks(range(len(results_df)))
ax4.set_xticklabels(results_df['Name'], rotation=45, ha='right')
ax4.set_ylabel('Cross-Validation Score')
ax4.set_title('Cross-Validation Performance (Mean ± Std)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Visualizations saved as 'experiment_results.png'")

# Save results to CSV
results_df.to_csv('experiment_results.csv', index=False)
print("✅ Results table saved as 'experiment_results.csv'")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
