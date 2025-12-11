"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.1.1
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def create_split_column(df_raw: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds a 'split' column to the dataframe with values 'train', 'test', 
    or 'evaluate' based on the provided ratios.

    Args:
        df: Input pandas DataFrame.
        parameters: Dictionary containing split ratios and random seed.
                    Expected keys: 'train_ratio', 'test_ratio', 'eval_ratio', 'seed'.

    Returns:
        pd.DataFrame: DataFrame with the new 'split' column.
    """
    
    # 1. Extract parameters
    split_probs = [params["train_ratio"], params["test_ratio"]]
    split_labels = ['train', 'test']
    base_seed = params.get("seed", 42)

    # 2. Validate that ratios sum to 1.0 (approx)
    if not np.isclose(sum(split_probs), 1.0):
        raise ValueError(f"Split ratios must sum to 1. Current sum: {sum(split_probs)}")

    # 3. Set random seed for reproducibility
    np.random.seed(base_seed)

    # 4. Create folds column
    df_out = df_raw.copy()

    # 3. Apply the split using groupby and transform
    # We use 'transform' to return a sequence of the same length as the group
    df_out["split"] = np.random.choice(split_labels, size=len(df_out), p=split_probs)

    return df_out


def split_master_table(df_mt, columns) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df_mt.loc[df_mt['split'] == 'train'].copy()
    df_test = df_mt.loc[df_mt['split'] == 'test'].copy()

    return df_train, df_test


def encoder_fit(df_in: pd.DataFrame, params: Dict[str, Any]) -> LabelEncoder:
    from sklearn.preprocessing import LabelEncoder
    
    encoder = LabelEncoder()

    encoder.fit(df_in[params['target']])

    return encoder


def encoder_transform(df_in: pd.DataFrame, encoder: LabelEncoder, params: Dict[str, Any]) -> pd.DataFrame:
    
    df_out = df_in.copy()

    df_out[params['target'] + '_encoded'] = encoder.transform(df_in[params['target']])


    return df_out


def scaler_fit(df_in: pd.DataFrame, params: Dict[str, Any]) -> StandardScaler:
    """
    The function `scaler_fit` fits a StandardScaler to the specified features in a DataFrame.
    
    Args:
      df_in (pd.DataFrame): A pandas DataFrame containing the data to be scaled.
      params (Dict[str, Any]): params = {
    
    Returns:
      a trained `StandardScaler` object that has been fitted to the specified features in the input
    DataFrame according to the provided parameters.
    """
    
    scaler = StandardScaler()

    scaler.fit(df_in[params['features']])

    return scaler


def scaler_transform(df_in: pd.DataFrame, scaler: StandardScaler, columns: Dict[str, Any]) -> pd.DataFrame:
    """
    The function `scaler_transform` takes a DataFrame, a StandardScaler, and specified columns to scale
    the features and concatenate them back into the original DataFrame.
    
    Args:
      df_in (pd.DataFrame): `df_in` is a pandas DataFrame containing the input data that you want to
    transform using the scaler.
      scaler (StandardScaler): The `scaler` parameter in the `scaler_transform` function is expected to
    be an instance of the `StandardScaler` class from scikit-learn or any other library that implements
    the same interface for scaling numerical data. This scaler is used to transform the specified
    features in the input DataFrame (`
      columns (Dict[str, Any]): The `columns` parameter seems to be a dictionary with keys 'features'
    and potentially other keys. The 'features' key likely contains a list of column names that you want
    to scale using the provided `scaler`.
    
    Returns:
      The function `scaler_transform` returns a pandas DataFrame `df_out` which is the original input
    DataFrame `df_in` with the specified columns scaled using the provided `scaler` (StandardScaler) and
    concatenated back with the original DataFrame.
    """
    
    df_scaled = pd.DataFrame(scaler.transform(df_in[columns['features']]), index = df_in.index, columns = columns['features'])
    
    df_out = pd.concat([
            df_in.drop(columns = columns['features']),
            df_scaled
        ], axis = 1)

    return df_out