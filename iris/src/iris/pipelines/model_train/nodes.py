"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 1.1.1
"""
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from typing import Dict, Any

def train_model(df_mt: pd.DataFrame,  columns: Dict[str, Any],  params: Dict[str, Any]) -> BaseEstimator:
    from sklearn.linear_model import LogisticRegression

    X_train = df_mt.loc[df_mt['split'] == 'train', columns['features']]
    y_train = df_mt.loc[df_mt['split'] == 'train', columns['target'] + '_encoded']

    model = LogisticRegression().fit(X_train, y_train)

    return model

def predict_model(df_in: pd.DataFrame, model: BaseEstimator, columns: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
    df_input = df_in[columns['features']]
    
    predictions = model.predict(df_input)

    df_out = pd.DataFrame(predictions, index = df_input.index, columns = ['prediction'])

    return df_out


def evaluate_predictions(df_y: pd.DataFrame, df_y_hat, columns: Dict[str, Any], params: Dict[str, Any]) -> Dict:
    from sklearn.metrics import classification_report

    report = classification_report(df_y[columns['target'] + '_encoded'], df_y_hat['prediction'], output_dict = True)
    
    return report
