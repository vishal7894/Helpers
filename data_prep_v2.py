import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from warnings import filterwarnings
filterwarnings("ignore")

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)


def load_data(folder_path, file_name):
    """Load and perform initial data cleaning."""
    df = pd.read_excel(os.path.join(folder_path, file_name))
    print(f"Original data shape: {df.shape}")
    
    # Drop unwanted columns
    df.drop(columns=["ReferenceDate"], inplace=True)
    
    # Convert date columns
    df['ReferenceDate'] = pd.to_datetime(df['MonthKey'])
    
    # Filter for dates before today
    today = datetime.today().strftime('%Y-%m-%d')
    df = df[df["ReferenceDate"] <= today]
    
    # Extract date features
    df["ReferenceMonth"] = df['ReferenceDate'].dt.month
    df["ReferenceQuarter"] = df['ReferenceDate'].dt.quarter
    df["ReferenceYear"] = df['ReferenceDate'].dt.year
    df['LastDepositMonth'] = pd.to_datetime(df['LastDepositMonth'], errors='coerce')
    
    return df


def process_columns(df):
    """Process all columns efficiently with vectorized operations."""
    # Fill nulls for specific columns
    fill_zero_cols = ["PrepaidSpends", "DebitSpends"]
    fill_ffill_cols = ["LastDepositMonth", "Gender", "University", "Flag", "CustomerAge@PostDate"]
    
    # Create customer groups for efficient processing
    df_grouped = df.groupby('CustomerNumber')
    
    # Forward fill for specific columns grouped by customer
    for col in fill_ffill_cols:
        df[col] = df_grouped[col].transform(lambda x: x.ffill())
    
    # Fill zeros for spend columns
    for col in fill_zero_cols:
        df[col] = df[col].fillna(0)
    
    # Flag conversion
    df["Attrition_flag"] = np.where(df["Flag"] == "Active", 0, 1)
    
    # Calculate months since last deposit
    df['months_since_last_deposit'] = (df['ReferenceDate'].dt.year - df['LastDepositMonth'].dt.year) * 12 + \
                                       (df['ReferenceDate'].dt.month - df['LastDepositMonth'].dt.month)
    
    return df


def calculate_features(df):
    """Calculate all aggregated features using vectorized operations."""
    # Group data by customer for rolling calculations
    customers = df.groupby('CustomerNumber')
    
    # Balance Features
    df['last_3m_avg_balance'] = customers['AvgMonthlyBalance'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # Yearly and quarterly aggregates - Balance
    df["yearly_balance"] = df.groupby(['CustomerNumber', 'ReferenceYear'])["AvgMonthlyBalance"].transform('sum')
    df["quarterly_balance"] = df.groupby(['CustomerNumber', 'ReferenceYear', 'ReferenceQuarter'])["AvgMonthlyBalance"].transform('sum')
    
    # Monthly and quarterly indices - Balance
    # Add epsilon to avoid division by zero
    df["balance_monthly_index"] = df["AvgMonthlyBalance"] / (df["yearly_balance"] + 1e-10)
    df["balance_quarterly_index"] = df["AvgMonthlyBalance"] / (df["quarterly_balance"] + 1e-10)
    
    # Debit Spend Features
    df['last_3m_avg_d_spend'] = customers['DebitSpends'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # Yearly and quarterly aggregates - Debit Spend
    df["yearly_d_spend"] = df.groupby(['CustomerNumber', 'ReferenceYear'])["DebitSpends"].transform('sum')
    df["quarterly_d_spend"] = df.groupby(['CustomerNumber', 'ReferenceYear', 'ReferenceQuarter'])["DebitSpends"].transform('sum')
    
    # Monthly and quarterly indices - Debit Spend
    df["d_spend_monthly_index"] = df["DebitSpends"] / (df["yearly_d_spend"] + 1e-10)
    df["d_spend_quarterly_index"] = df["DebitSpends"] / (df["quarterly_d_spend"] + 1e-10)
    
    # Prepaid Spend Features
    df['last_3m_avg_p_spend'] = customers['PrepaidSpends'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # Yearly and quarterly aggregates - Prepaid Spend
    df["yearly_p_spend"] = df.groupby(['CustomerNumber', 'ReferenceYear'])["PrepaidSpends"].transform('sum')
    df["quarterly_p_spend"] = df.groupby(['CustomerNumber', 'ReferenceYear', 'ReferenceQuarter'])["PrepaidSpends"].transform('sum')
    
    # Monthly and quarterly indices - Prepaid Spend
    df["p_spend_monthly_index"] = df["PrepaidSpends"] / (df["yearly_p_spend"] + 1e-10)
    df["p_spend_quarterly_index"] = df["PrepaidSpends"] / (df["quarterly_p_spend"] + 1e-10)
    
    # Login Features
    df["yearly_logins"] = df.groupby(['CustomerNumber', 'ReferenceYear'])["LOGIN_COUNT_COL"].transform('sum')
    df["quarterly_logins"] = df.groupby(['CustomerNumber', 'ReferenceYear', 'ReferenceQuarter'])["LOGIN_COUNT_COL"].transform('sum')
    
    # Monthly and quarterly indices - Logins
    df["logins_monthly_index"] = df["LOGIN_COUNT_COL"] / (df["yearly_logins"] + 1e-10)
    df["logins_quarterly_index"] = df["LOGIN_COUNT_COL"] / (df["quarterly_logins"] + 1e-10)
    
    return df


def create_target_variable(df):
    """Create target variable by shifting attrition flags."""
    # Get all customers with at least 6 records
    valid_customers = df.groupby('CustomerNumber').size()
    valid_customers = valid_customers[valid_customers >= 6].index
    
    # Filter for valid customers
    df_valid = df[df['CustomerNumber'].isin(valid_customers)].copy()
    
    # Sort dataframe for correct shifting
    df_valid = df_valid.sort_values(['CustomerNumber', 'ReferenceDate'])
    
    # Group by customer and create shifted target
    df_valid['target'] = df_valid.groupby('CustomerNumber')['Attrition_flag'].shift(-2)
    
    return df_valid


def split_train_test(df):
    """Split data into train and test sets based on date criteria."""
    # Create model dataframe
    df_model = df.copy()
    
    # Create training dataframe excluding rows with NA target
    train_agg_df = df[~df['target'].isna()].copy()
    
    # Split into train and test based on date
    customer_max_dates = train_agg_df.groupby('CustomerNumber')['ReferenceDate'].max()
    
    train_test_splits = {}
    for cust_id, max_date in customer_max_dates.items():
        cutoff_date = max_date - relativedelta(months=2) + pd.offsets.MonthEnd(0)
        customer_data = train_agg_df[train_agg_df['CustomerNumber'] == cust_id]
        
        train_data = customer_data[customer_data['ReferenceDate'] <= cutoff_date]
        test_data = customer_data[customer_data['ReferenceDate'] > cutoff_date]
        
        train_test_splits[cust_id] = (train_data, test_data)
    
    # Combine all train and test data
    train_dfs = [t[0] for t in train_test_splits.values() if not t[0].empty]
    test_dfs = [t[1] for t in train_test_splits.values() if not t[1].empty]
    
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
    
    return df_model, train_agg_df, train_df, test_df


def prep_data_optimized(df, MAX_DATE):
    """Optimized version of prep_data function with vectorized operations."""
    # Process only customer records up to MAX_DATE
    df = df[df['ReferenceDate'] <= MAX_DATE].copy()
    
    print("Processing columns...")
    df = process_columns(df)
    
    print("Calculating features...")
    df = calculate_features(df)
    
    print("Creating target variable...")
    df_valid = create_target_variable(df)
    
    print("Splitting train/test data...")
    df_model, train_agg_df, train_df, test_df = split_train_test(df_valid)
    
    print(f"Final shapes: model={df_model.shape}, train_agg={train_agg_df.shape}, "
          f"train={train_df.shape}, test={test_df.shape}")
    
    return df_model, train_agg_df, train_df, test_df


if __name__ == "__main__":
    # Define file paths
    folder_path = r"D:\Attrition\Data\Raw"
    file_name = r"sample_raw.xlsx"
    
    # Load and preprocess data
    print("Loading data...")
    df = load_data(folder_path, file_name)
    
    # Run optimized data preparation
    print("Running optimized data preparation...")
    import time
    start_time = time.time()
    
    df_model, train_agg_df, train_df, test_df = prep_data_optimized(df, df["ReferenceDate"].max())
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")