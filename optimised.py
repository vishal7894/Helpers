import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import concurrent.futures
from tqdm import tqdm

def fill_null_with_ffill(df, columns):
    """Fill null values with forward fill for specific columns."""
    return df.assign(**{col: df[col].ffill() for col in columns})

def fill_null_with_zero(df, columns):
    """Fill null values with zero for specific columns."""
    return df.assign(**{col: df[col].fillna(0) for col in columns})

def process_customer(group_tuple, MAX_DATE):
    """Process a single customer's data."""
    cust_id, df_slice = group_tuple
    
    # Get first valid index and min allowance date
    idx = df_slice["LastDepositMonth"].first_valid_index()
    min_allowance_date = df_slice["LastDepositMonth"].min()
    
    # Skip if conditions aren't met
    if df_slice["ReferenceDate"].max() != MAX_DATE or not idx:
        return None, None, None, None
    
    # Apply initial filters
    df_slice = df_slice.loc[idx:]
    df_slice = df_slice[df_slice["ReferenceDate"] >= min_allowance_date]
    
    # Skip if not enough data after filtering
    if df_slice.shape[0] < 6:
        return None, None, None, None
    
    # Sort by date at the beginning to ensure consistent ordering
    df_slice = df_slice.sort_values(by="ReferenceDate")
    
    # Fill nulls
    df_slice = fill_null_with_ffill(df_slice, ["LastDepositMonth", "Gender", "University", "Flag", "CustomerAge@PostDate"])
    df_slice = fill_null_with_zero(df_slice, ["PrepaidSpends", "DebitSpends"])
    
    # Flag conversion - vectorized
    df_slice["Attrition_flag"] = (df_slice["Flag"] != "Active").astype(int)
    
    # Date difference calculation - vectorized
    df_slice['months_since_last_deposit'] = ((df_slice['ReferenceDate'].dt.year - df_slice['LastDepositMonth'].dt.year) * 12 + 
                                           (df_slice['ReferenceDate'].dt.month - df_slice['LastDepositMonth'].dt.month))
    
    # Generate features - vectorized where possible
    # Rolling averages
    df_slice['last_3m_avg_balance'] = df_slice['AvgMonthlyBalance'].rolling(window=3, min_periods=1).mean()
    df_slice['last_3m_avg_d_spend'] = df_slice['DebitSpends'].rolling(window=3, min_periods=1).mean()
    df_slice['last_3m_avg_p_spend'] = df_slice['PrepaidSpends'].rolling(window=3, min_periods=1).mean()
    
    # Group aggregations - more efficient grouped operations
    for period, groupby_cols in [("yearly", ["ReferenceYear"]), ("quarterly", ["ReferenceYear", "ReferenceQuarter"])]:
        # Balance metrics
        df_slice[f"{period}_balance"] = df_slice.groupby(groupby_cols)["AvgMonthlyBalance"].transform("sum")
        df_slice[f"balance_{period}_index"] = df_slice["AvgMonthlyBalance"] / df_slice[f"{period}_balance"].replace(0, np.nan)
        
        # Debit spend metrics
        df_slice[f"{period}_d_spend"] = df_slice.groupby(groupby_cols)["DebitSpends"].transform("sum")
        df_slice[f"d_spend_{period}_index"] = df_slice["DebitSpends"] / df_slice[f"{period}_d_spend"].replace(0, np.nan)
        
        # Prepaid spend metrics
        df_slice[f"{period}_p_spend"] = df_slice.groupby(groupby_cols)["PrepaidSpends"].transform("sum")
        df_slice[f"p_spend_{period}_index"] = df_slice["PrepaidSpends"] / df_slice[f"{period}_p_spend"].replace(0, np.nan)
        
        # Login metrics
        df_slice[f"{period}_logins"] = df_slice.groupby(groupby_cols)["LOGIN_COUNT_COL"].transform("sum")
        df_slice[f"logins_{period}_index"] = df_slice["LOGIN_COUNT_COL"] / df_slice[f"{period}_logins"].replace(0, np.nan)
    
    # Create target variable - more efficiently
    df_slice["target"] = np.nan
    df_slice.iloc[:-2, df_slice.columns.get_loc("target")] = df_slice["Attrition_flag"].iloc[2:].values
    
    # Split into train and test sets
    cutoff_date = df_slice["ReferenceDate"].max() - relativedelta(months=2) + pd.offsets.MonthEnd(0)
    Train_df = df_slice[~df_slice["target"].isna()]
    train_df = Train_df[Train_df["ReferenceDate"] <= cutoff_date]
    test_df = Train_df[Train_df["ReferenceDate"] > cutoff_date]
    
    return df_slice, Train_df, train_df, test_df

def prep_data_optimized(df, MAX_DATE, n_workers=4):
    """Optimized data preparation function using parallel processing."""
    print(f"Processing {df['CustomerNumber'].nunique()} unique customers with {n_workers} workers")
    
    # Pre-compute ReferenceYear and ReferenceQuarter to avoid redundant calculations
    df['ReferenceYear'] = df['ReferenceDate'].dt.year
    df['ReferenceQuarter'] = df['ReferenceDate'].dt.quarter
    
    # Group data by customer ID first (outside the loop)
    grouped = list(df.groupby('CustomerNumber'))
    
    # Initialize result DataFrames
    df_model = pd.DataFrame()
    Train_agg_df = pd.DataFrame()
    train_agg_df = pd.DataFrame()
    test_agg_df = pd.DataFrame()
    
    # Process customers in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Map the processing function to each customer group
        results = list(tqdm(
            executor.map(lambda x: process_customer(x, MAX_DATE), grouped),
            total=len(grouped),
            desc="Processing customers"
        ))
    
    # Combine results
    for df_slice, Train_df, train_df, test_df in results:
        if df_slice is not None:
            df_model = pd.concat([df_model, df_slice], ignore_index=True)
        if Train_df is not None:
            Train_agg_df = pd.concat([Train_agg_df, Train_df], ignore_index=True)
        if train_df is not None:
            train_agg_df = pd.concat([train_agg_df, train_df], ignore_index=True)
        if test_df is not None:
            test_agg_df = pd.concat([test_agg_df, test_df], ignore_index=True)
    
    return df_model, Train_agg_df, train_agg_df, test_agg_df

# # Run optimized function with 8 workers (adjust based on your CPU)
# df_model, Train_agg_df, train_agg_df, test_agg_df = prep_data_optimized(df, MAX_DATE, n_workers=8)
