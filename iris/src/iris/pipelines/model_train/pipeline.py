"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import train_model, evaluate_predictions
from ..data_engineering.nodes import split_master_table
from ..model_predict.nodes import predict_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func = split_master_table,
            inputs = ['master_table', 'params:columns'],
            outputs = ['mt_train', 'mt_test'],
            name = 'split_master_table'
        ),
        Node(
            func = train_model,
            inputs = ['mt_train', 'params:columns', 'params:train'],
            outputs = 'trained_model',
            name = 'train_model'
        ),
        Node(
            func = predict_model,
            inputs = ['mt_test', 'trained_model', 'params:columns', 'params:test'],
            outputs = 'test_predictions',
            name = 'predict_test'
        ),
        Node(
            func = evaluate_predictions,
            inputs = ['mt_test', 'test_predictions', 'params:columns', 'params:evaluate'],
            outputs = 'test_classification_report',
            name = 'evaluate_test'
        )
    ])