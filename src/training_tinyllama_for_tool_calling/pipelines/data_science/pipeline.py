from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_base_model  # evaluate_base_model,
from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table_df", "params:model_options"],
                outputs=[
                    "chat_threads_train_ds",
                    "chat_threads_validation_ds",
                    "chat_threads_test_ds",
                ],
                name="split_data_node",
            ),
            node(
                func=prepare_base_model,
                inputs=[
                    "chat_threads_train_ds",
                    "chat_threads_validation_ds",
                    "chat_threads_test_ds",
                ],
                outputs="pretrained_model_uri",
                name="prepare_base_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "pretrained_model_uri",
                    "chat_threads_test_ds",
                ],
                outputs="pretrained_model_evaluation_df",
                name="evaluate_base_model_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "chat_threads_train_ds",
                    "chat_threads_validation_ds",
                    "chat_threads_test_ds",
                    "params:model_config",
                    "params:sft_config",
                    "params:sft_script_arguments",
                    "pretrained_model_uri",
                ],
                outputs="tuned_model_uri",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "tuned_model_uri",
                    "chat_threads_test_ds",
                ],
                outputs="tuned_model_evaluation_df",
                name="evaluate_tuned_model_node",
            ),
        ]
    )
