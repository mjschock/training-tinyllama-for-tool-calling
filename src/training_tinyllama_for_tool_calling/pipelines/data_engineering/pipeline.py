from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table_df,
    preprocess_chat_threads_df,
    preprocess_drone_training_df,
    preprocess_gorilla_berkeley_function_call_leaderboard_v3_df,
    preprocess_llamafactory_glaive_toolcall_en_df,
    preprocess_nexusflow_nexusraven_api_evaluation_df,
    preprocess_nousresearch_hermes_function_calling_v1_df,
    preprocess_team_ace_toolace_df,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_chat_threads_df,
                inputs="chat_threads_df",
                outputs="preprocessed_chat_threads_df",
                name="preprocess_chat_threads_df_node",
            ),
            node(
                func=preprocess_drone_training_df,
                inputs="drone_training_df",
                outputs="preprocessed_drone_training_df",
                name="preprocess_drone_training_df_node",
            ),
            node(
                func=preprocess_gorilla_berkeley_function_call_leaderboard_v3_df,
                inputs="gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df",
                outputs="preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df",
                name="preprocess_gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df_node",
            ),
            node(
                func=preprocess_gorilla_berkeley_function_call_leaderboard_v3_df,
                inputs="gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df",
                outputs="preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df",
                name="preprocess_gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df_node",
            ),
            node(
                func=preprocess_gorilla_berkeley_function_call_leaderboard_v3_df,
                inputs="gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df",
                outputs="preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df",
                name="preprocess_gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df_node",
            ),
            node(
                func=preprocess_gorilla_berkeley_function_call_leaderboard_v3_df,
                inputs="gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df",
                outputs="preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df",
                name="preprocess_gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df_node",
            ),
            node(
                func=preprocess_llamafactory_glaive_toolcall_en_df,
                inputs="llamafactory_glaive_toolcall_en_df",
                outputs="preprocessed_llamafactory_glaive_toolcall_en_df",
                name="preprocess_llamafactory_glaive_toolcall_en_df_node",
            ),
            node(
                func=create_model_input_table_df,
                inputs=[
                    # "chat_threads_df",
                    "preprocessed_chat_threads_df",
                    "preprocessed_drone_training_df",
                    "preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df",
                    "preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df",
                    "preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df",
                    "preprocessed_gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df",
                    "preprocessed_llamafactory_glaive_toolcall_en_df",
                ],
                outputs="model_input_table_df",
                name="create_model_input_table_df_node",
            ),
        ]
    )
