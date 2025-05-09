import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score, average_precision_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')

# Features to use in all models
features = ["University_enc", "Gender_enc", "CustomerAge@PostDate", "months_since_last_deposit", 
            "last_3m_avg_balance", "balance_monthly_index", "balance_quarterly_index", 
            "last_3m_avg_d_spend", "d_spend_monthly_index", "d_spend_quarterly_index", 
            "last_3m_avg_p_spend", "p_spend_monthly_index", "p_spend_quarterly_index", 
            "logins_monthly_index", "logins_quarterly_index"]

def preprocess_data(train_agg_df, test_agg_df, Train_agg_df=None, test_df=None):
    """
    Encodes categorical features across all datasets
    """
    # Create label encoders
    le_uni = LabelEncoder()
    le_gender = LabelEncoder()
    
    # Fit encoders on all available data for consistent encoding
    all_universities = pd.concat([
        train_agg_df['University'], 
        test_agg_df['University'],
        Train_agg_df['University'] if Train_agg_df is not None else pd.Series(),
        test_df['University'] if test_df is not None else pd.Series()
    ])
    
    all_genders = pd.concat([
        train_agg_df['Gender'], 
        test_agg_df['Gender'],
        Train_agg_df['Gender'] if Train_agg_df is not None else pd.Series(),
        test_df['Gender'] if test_df is not None else pd.Series()
    ])
    
    le_uni.fit(all_universities)
    le_gender.fit(all_genders)
    
    # Transform each dataset
    if Train_agg_df is not None:
        Train_agg_df['University_enc'] = le_uni.transform(Train_agg_df['University'])
        Train_agg_df['Gender_enc'] = le_gender.transform(Train_agg_df['Gender'])

    if test_df is not None:
        test_df['University_enc'] = le_uni.transform(test_df['University'])
        test_df['Gender_enc'] = le_gender.transform(test_df['Gender'])
    
    train_agg_df['University_enc'] = le_uni.transform(train_agg_df['University'])
    train_agg_df['Gender_enc'] = le_gender.transform(train_agg_df['Gender'])
    
    test_agg_df['University_enc'] = le_uni.transform(test_agg_df['University'])
    test_agg_df['Gender_enc'] = le_gender.transform(test_agg_df['Gender'])
    
    return train_agg_df, test_agg_df, Train_agg_df, test_df, le_uni, le_gender

def get_models(class_weights=None):
    """
    Returns a dictionary of models to train and evaluate with class weight adjustment
    
    Args:
        class_weights: Dictionary mapping class indices to weights, e.g. {0: 1, 1: 5}
                      to address class imbalance
    """
    models = {
        'LGBM': lgb.LGBMClassifier(
            random_state=42, 
            n_estimators=100,
            class_weight='balanced' if class_weights is None else class_weights,
            scale_pos_weight=class_weights.get(1, 1)/class_weights.get(0, 1) if class_weights else None
        ),
        'XGBoost': XGBClassifier(
            random_state=42, 
            n_estimators=100,
            scale_pos_weight=class_weights.get(1, 1)/class_weights.get(0, 1) if class_weights else 1
        ),
        'RandomForest': RandomForestClassifier(
            random_state=42, 
            n_estimators=100,
            class_weight='balanced' if class_weights is None else class_weights
        ),
        'GradientBoosting': GradientBoostingClassifier(
            random_state=42, 
            n_estimators=100
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced' if class_weights is None else class_weights
        ),
        'CatBoost': CatBoostClassifier(
            random_state=42, 
            n_estimators=100, 
            verbose=0,
            auto_class_weights='Balanced' if class_weights is None else None,
            class_weights=list(class_weights.values()) if class_weights else None
        )
    }
    return models

def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, cutoff=0.5):
    """
    Trains a model and evaluates its performance
    """
    print(f"\n{'='*20} Training {model_name} {'='*20}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= cutoff).astype(int)
    
    # Calculate metrics for imbalanced classification
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)  # PR AUC is better for imbalanced data
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Print reports
    print(f"Model: {model_name}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"PR AUC (Average Precision): {pr_auc:.4f}") # Better for imbalanced data
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(
        y_test,
        y_proba,
        name=f"{model_name}",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()
    
    # Plot PR curve for imbalanced data
    plt.figure(figsize=(10, 8))
    from sklearn.metrics import PrecisionRecallDisplay
    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_proba,
        name=f"{model_name}",
        color="darkorange",
    )
    plt.axhline(y=sum(y_test)/len(y_test), color="k", linestyle="--", 
                label=f"No Skill (AP = {sum(y_test)/len(y_test):.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.show()
    
    # Plot feature importance if available
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        if model_name == 'LGBM':
            lgb.plot_importance(model, max_num_features=15)
            plt.title(f"{model_name} Feature Importance")
        else:
            # Generic feature importance plot for other models
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            plt.bar(range(len(indices[:15])), importance[indices[:15]])
            plt.xticks(range(len(indices[:15])), [features[i] for i in indices[:15]], rotation=90)
            plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    
    # Create prediction dataframe
    prediction_df = pd.DataFrame()
    if 'ReferenceDate' in X_test.columns and 'Customernumber' in X_test.columns:
        prediction_df = X_test[['ReferenceDate', 'Customernumber']].copy()
    
    prediction_df['prediction'] = y_pred
    prediction_df['probability'] = y_proba
    
    # Return model, metrics, and predictions
    return {
        'model': model,
        'model_name': model_name,
        'auc': auc,
        'pr_auc': pr_auc,  # Added PR AUC
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'prediction_df': prediction_df
    }

def analyze_class_imbalance(y):
    """
    Analyzes class imbalance and provides visualizations and statistics
    """
    class_counts = np.bincount(y)
    class_percentages = 100 * class_counts / len(y)
    
    print(f"\n{'='*20} Class Imbalance Analysis {'='*20}")
    print(f"Class 0: {class_counts[0]} samples ({class_percentages[0]:.2f}%)")
    print(f"Class 1: {class_counts[1]} samples ({class_percentages[1]:.2f}%)")
    print(f"Imbalance ratio (majority:minority): {max(class_counts)/min(class_counts):.2f}:1")
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=['Class 0', 'Class 1'], y=class_counts)
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{class_counts[i]}\n({class_percentages[i]:.2f}%)", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom')
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.show()
    
    # Calculate appropriate class weights
    if class_counts[0] > class_counts[1]:
        imbalance_ratio = class_counts[0] / class_counts[1]
        class_weights = {0: 1, 1: imbalance_ratio}
    else:
        imbalance_ratio = class_counts[1] / class_counts[0]
        class_weights = {0: imbalance_ratio, 1: 1}
    
    return class_weights

def apply_resampling(X, y, method='smote', sampling_strategy=0.5):
    """
    Applies various resampling techniques to address class imbalance
    
    Args:
        X: Features
        y: Target
        method: Resampling method ('smote', 'adasyn', 'undersample', 'smotetomek', 'smoteenn')
        sampling_strategy: For oversampling, fraction of majority class to oversample minority class to
                          For undersampling, fraction of majority class to keep
    
    Returns:
        X_resampled, y_resampled: Resampled data
    """
    print(f"\nApplying {method.upper()} resampling...")
    orig_shape = X.shape
    
    if method.lower() == 'smote':
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method.lower() == 'adasyn':
        resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    elif method.lower() == 'undersample':
        resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif method.lower() == 'smotetomek':
        resampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
    elif method.lower() == 'smoteenn':
        resampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    else:
        print(f"Warning: Unknown resampling method '{method}'. Using original data.")
        return X, y
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    # Print resampling stats
    print(f"Original shape: {orig_shape}")
    print(f"Resampled shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled

def train_and_evaluate_multiple_models(X_train, y_train, X_test, y_test, cutoff=0.5, 
                                      metric='pr_auc', handle_imbalance='auto', 
                                      resampling_method=None):
    """
    Trains and evaluates multiple models, returns results dictionary with metrics
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        cutoff: Classification threshold
        metric: Metric to use for model selection ('auc', 'pr_auc', 'f1', 'precision', 'recall')
        handle_imbalance: How to handle class imbalance
                         'auto': Use class weights
                         'none': Do nothing
                         'resampling': Use resampling method specified in resampling_method
        resampling_method: If handle_imbalance='resampling', which method to use
                          ('smote', 'adasyn', 'undersample', 'smotetomek', 'smoteenn')
    """
    # Analyze class imbalance
    class_weights = analyze_class_imbalance(y_train)
    
    # Handle class imbalance
    if handle_imbalance == 'auto':
        print("\nHandling class imbalance with class weights...")
        models = get_models(class_weights)
        X_train_res, y_train_res = X_train, y_train  # No resampling, just use weights
    elif handle_imbalance == 'resampling' and resampling_method is not None:
        print(f"\nHandling class imbalance with {resampling_method} resampling...")
        X_train_res, y_train_res = apply_resampling(X_train, y_train, 
                                                  method=resampling_method)
        models = get_models()  # No need for class weights with resampling
    else:
        print("\nNo class imbalance handling applied.")
        models = get_models()
        X_train_res, y_train_res = X_train, y_train
    
    results = {}
    
    for model_name, model in models.items():
        try:
            model_results = train_evaluate_model(
                model, model_name, X_train_res, y_train_res, X_test, y_test, cutoff
            )
            results[model_name] = model_results
        except Exception as e:
            print(f"Error training model {model_name}: {str(e)}")
    
    # Find best model based on specified metric
    best_model_name = max(results, key=lambda x: results[x][metric])
    best_model_score = results[best_model_name][metric]
    
    print(f"\n{'='*20} Results Summary {'='*20}")
    print(f"Best model based on {metric}: {best_model_name} with {metric} = {best_model_score:.4f}")
    
    # Display all model metrics
    metrics_df = pd.DataFrame({
        'Model': [name for name in results.keys()],
        'AUC': [results[name]['auc'] for name in results.keys()],
        'PR_AUC': [results[name]['pr_auc'] for name in results.keys()],  # Added PR AUC
        'F1': [results[name]['f1'] for name in results.keys()],
        'Precision': [results[name]['precision'] for name in results.keys()],
        'Recall': [results[name]['recall'] for name in results.keys()]
    })
    
    column_map = {'auc': 'AUC', 'pr_auc': 'PR_AUC', 'f1': 'F1', 
                 'precision': 'Precision', 'recall': 'Recall'}
    metrics_df = metrics_df.sort_values(by=column_map.get(metric, 'AUC'), ascending=False)
    print("\nMetrics for all models:")
    print(metrics_df)
    
    # Save the best model
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, f'best_model_{best_model_name}.pkl')
    print(f"Best model saved as 'best_model_{best_model_name}.pkl'")
    
    return results, best_model_name

def predict_future(X_future, model_path=None, model=None, cutoff=0.5):
    """
    Makes predictions on future data using either a saved model or a model object
    """
    if model is None and model_path is not None:
        # Load the saved model
        model = joblib.load(model_path)
    elif model is None and model_path is None:
        raise ValueError("Either model or model_path must be provided")
    
    # Predict probabilities
    y_proba = model.predict_proba(X_future[features])[:, 1]
    
    # Apply custom cutoff
    y_pred = (y_proba >= cutoff).astype(int)
    
    # Create prediction dataframe
    prediction_df = pd.DataFrame()
    if 'ReferenceDate' in X_future.columns and 'Customernumber' in X_future.columns:
        prediction_df = X_future[['ReferenceDate', 'Customernumber']].copy()
    
    prediction_df['prediction'] = y_pred
    prediction_df['probability'] = y_proba
    prediction_df['CustomerNumber'] = X_future["CustomerNumber"]
    
    return prediction_df

# Example of how to use these functions:
# Preprocess data
train_df, test_df, train_agg_df, test_agg_df, le_uni, le_gender = preprocess_data(train_df, test_df, train_agg_df, test_agg_df)

# Prepare features and targets
x_train, y_train = train_df[features], train_df['target']
x_test, y_test = test_df[features], test_df['target']
X_train, Y_train = train_agg_df[features], train_agg_df['target'] if train_agg_df is not None else (None, None)
X_test, Y_test = test_agg_df[features], test_agg_df['target'] if test_agg_df is not None else (None, None)

# Train and evaluate multiple models
results, best_model_name = train_and_evaluate_multiple_models(
    x_train, y_train, x_test, y_test, cutoff=0.5, metric='auc'
)

# Make predictions with the best model
best_model = results[best_model_name]['model']
final_predictions = predict_future(X_train, model=best_model, cutoff=0.5)
