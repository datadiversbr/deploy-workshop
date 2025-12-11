"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import create_split_column, split_master_table, encoder_fit, encoder_transform, scaler_fit, scaler_transform

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func = create_split_column,
            inputs = ['iris_train_raw', 'params:split_params'],
            outputs = 'iris_split',
            name = 'create_split_column'
        ),
        Node(
            func = split_master_table,
            inputs = ['iris_split', 'params:columns'],
            outputs = ['train_raw', 'test_raw'],
            name = 'split_raw_table'
        ),
        Node(
            func = scaler_fit,
            inputs = ['train_raw', 'params:columns'],
            outputs = 'feature_scaler_fitted',
            name = 'create_feature_scaler'
        ),
        Node(
            func = encoder_fit,
            inputs = ['train_raw', 'params:columns'],
            outputs = 'target_encoder_fitted',
            name = 'create_target_encoder'
        ),
        Node(
            func = scaler_transform,
            inputs = ['iris_split', 'feature_scaler_fitted', 'params:columns'],
            outputs = 'iris_scaled',
            name = 'scaled_master_table'
        ),

        Node(
            func = encoder_transform,
            inputs = ['iris_scaled', 'target_encoder_fitted', 'params:columns'],
            outputs = 'master_table',
            name = 'encode_target'
        )
    ])
