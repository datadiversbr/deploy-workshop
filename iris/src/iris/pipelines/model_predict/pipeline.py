"""
This is a boilerplate pipeline 'data_analysis'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from ..data_engineering.nodes import scaler_transform, encoder_transform
from .nodes import predict_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func = scaler_transform,
            inputs = ['iris_predict_raw', 'feature_scaler_fitted', 'params:columns'],
            outputs = 'predict_master_table',
            name = 'scale_prediction_table'
        ),
        Node(
            func = predict_model,
            inputs = ['predict_master_table', 'trained_model', 'params:columns', 'params:prediction'],
            outputs = 'predicted',
            name = 'create_predicted'
        ),
    ])
