
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union

def column_wise_clustering(df: pd.DataFrame, 
                          columns_to_cluster: List[str], 
                          n_components: int = 3,
                          random_state: int = 42) -> Dict[str, Any]:
    """
    Performs clustering on each specified column separately and segments customers.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe containing customer data
    columns_to_cluster : list of str
        Column names to perform clustering on
    n_components : int, default=3
        Number of clusters to create for each column
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'segmented_df': DataFrame with original data and cluster assignments
        - 'cluster_stats': Statistics for each cluster by column
        - 'kmeans_models': Fitted KMeans models for each column
    """
    # Initialize storage for results
    results = {}
    segmented_df = df.copy()
    cluster_stats = {}
    kmeans_models = {}
    
    # Process each column
    for col in columns_to_cluster:
        # Skip if column doesn't exist
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
            
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' is not numeric. Skipping.")
            continue
        
        # Handle missing values for this column
        col_data = df[col].copy()
        if col_data.isna().any():
            print(f"Warning: Column '{col}' contains missing values. Imputing with mean.")
            col_data = col_data.fillna(col_data.mean())
        
        # Reshape for clustering
        X = col_data.values.reshape(-1, 1)
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_components, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Store cluster assignments
        cluster_col_name = f"{col}_cluster"
        segmented_df[cluster_col_name] = cluster_labels
        
        # Calculate statistics for each cluster
        cluster_info = {}
        for cluster_id in range(n_components):
            mask = cluster_labels == cluster_id
            cluster_info[cluster_id] = {
                'count': np.sum(mask),
                'percentage': 100 * np.sum(mask) / len(df),
                'mean': df.loc[mask, col].mean(),
                'median': df.loc[mask, col].median(),
                'min': df.loc[mask, col].min(),
                'max': df.loc[mask, col].max(),
                'std': df.loc[mask, col].std()
            }
        
        # Store results
        cluster_stats[col] = cluster_info
        kmeans_models[col] = kmeans
    
    # Compile results
    results['segmented_df'] = segmented_df
    results['cluster_stats'] = cluster_stats
    results['kmeans_models'] = kmeans_models
    
    return results
