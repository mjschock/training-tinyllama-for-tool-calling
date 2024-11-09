import json
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import pandas as pd
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)


def _extract_function_calls(call_list):
    # This regex captures the function name and the arguments
    pattern = r"(\w+)\((.*)\)"

    extracted_calls = []

    for call in call_list:
        match = re.match(pattern, call)

        if match:
            func_name = match.group(1)
            args = match.group(2)
            # Split the arguments by commas, handling nested tuples
            arg_list = []
            nested_level = 0
            current_arg = []

            for char in args:
                if char == "(":
                    nested_level += 1

                elif char == ")":
                    nested_level -= 1

                if char == "," and nested_level == 0:
                    arg_list.append("".join(current_arg).strip())
                    current_arg = []

                else:
                    current_arg.append(char)

            # Add the last argument if exists
            if current_arg:
                arg_list.append("".join(current_arg).strip())

            extracted_calls.append((func_name, arg_list))

    # Verification step
    for func_name, args in extracted_calls:
        reconstructed_call = f"{func_name}({', '.join(args)})"

        if reconstructed_call not in call_list:
            print(
                f"Warning: Cannot recreate original string for {func_name} with args {args}"
            )
            raise Exception("Extraction failed")

    return extracted_calls


def _parse_chat_threads(
    gorilla_berkeley_function_call_leaderboard_v3_df: pd.DataFrame,
) -> pd.DataFrame:
    chat_threads = []

    for index, row in gorilla_berkeley_function_call_leaderboard_v3_df.iterrows():
        execution_result_type = row["execution_result_type"]

        # If any of the execution results are not exact matches, skip the example
        if any([x != "exact_match" for x in execution_result_type]):
            print("execution_result_type")
            print(execution_result_type)

            print("Skipping...")
            continue

        question = row["question"]
        assert len(question) == 1, f"{len(question)} != 1"

        ground_truth = row["ground_truth"]

        tool_calls = []

        try:
            signatures = _extract_function_calls(ground_truth)

        except Exception as e:
            print(e)
            print("Skipping...")
            continue

        for signature in signatures:
            tool_calls.append(
                {
                    "function": {
                        "arguments": signature[1],
                        "name": signature[0],
                    },
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                }
            )

        messages = question[0] + [
            {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
        ]

        function = row["function"]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in function
        ]

        chat_threads.append(
            {
                "documents": [],
                "messages": messages,
                "tools": tools,
            }
        )

    return pd.DataFrame(chat_threads)


def preprocess_drone_training_df(drone_training_df: pd.DataFrame) -> pd.DataFrame:
    drone_training_df["documents"] = pd.Series(
        [[] for _ in range(len(drone_training_df))]
    )

    drone_training_df = drone_training_df.drop(
        columns=["parallel_tool_calls"], inplace=False
    )

    return drone_training_df


def preprocess_gorilla_berkeley_function_call_leaderboard_v3_df(
    gorilla_berkeley_function_call_leaderboard_v3_df: pd.DataFrame,
) -> pd.DataFrame:

    gorilla_berkeley_function_call_leaderboard_v3_chat_threads_df = _parse_chat_threads(
        gorilla_berkeley_function_call_leaderboard_v3_df
    )

    return gorilla_berkeley_function_call_leaderboard_v3_chat_threads_df


@dataclass
class Message:
    content: Optional[
        Union[
            str,
            List[
                ChatCompletionContentPartTextParam
                | ChatCompletionContentPartImageParam
                | ChatCompletionContentPartInputAudioParam
            ],
        ]
    ] = None
    name: Optional[str] = None
    role: Literal["assistant", "system", "user", "tool"] = "user"
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


@dataclass
class Tool:
    function: Optional[dict] = None
    name: Optional[str] = None
    type: Literal["function"] = "function"


def _convert_json_list_to_string(json_list):
    if not isinstance(json_list, list):
        json_list = []

    if json_list:
        tool_call_initial_id_to_tool_call_id = {}

        for json_item in json_list:
            try:
                message = Message(**json_item)

                if message.tool_calls:
                    for message_tool_call in message.tool_calls:
                        tool_call_initial_id_to_tool_call_id[
                            message_tool_call["id"]
                        ] = f"call_{len(tool_call_initial_id_to_tool_call_id)}"

                        message_tool_call["id"] = tool_call_initial_id_to_tool_call_id[
                            message_tool_call["id"]
                        ]

                if message.role == "tool":
                    message.tool_call_id = tool_call_initial_id_to_tool_call_id[
                        message.tool_call_id
                    ]

                    json_item["tool_call_id"] = message.tool_call_id

            except TypeError as e:
                tool = Tool(**json_item)

            except TypeError:
                raise Exception("Invalid JSON")

    return json.dumps(json_list)


def create_model_input_table_df(
    chat_threads_df: pd.DataFrame,
    drone_training_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df: pd.DataFrame,
) -> pd.DataFrame:
    model_input_table = pd.concat(
        [
            chat_threads_df,
            drone_training_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df,
        ],
        ignore_index=True,
    )

    model_input_table["documents"] = model_input_table["documents"].apply(
        _convert_json_list_to_string
    )
    model_input_table["messages"] = model_input_table["messages"].apply(
        _convert_json_list_to_string
    )
    model_input_table["tools"] = model_input_table["tools"].apply(
        _convert_json_list_to_string
    )

    return model_input_table
