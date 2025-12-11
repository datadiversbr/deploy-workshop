"""
This is a boilerplate pipeline 'model_predict'
generated using Kedro 1.1.1
"""
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from typing import Dict, Any

def predict_model(df_in: pd.DataFrame, model: BaseEstimator, columns: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
    df_input = df_in[columns['features']]
    
    predictions = model.predict(df_input)

    df_out = pd.DataFrame(predictions, index = df_input.index, columns = ['prediction'])

    return df_out