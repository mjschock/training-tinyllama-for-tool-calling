import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from datasets import DatasetDict
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceLogProbs,
    ChatMessage,
    ChatParams,
    ChatResponse,
    FunctionToolCallArguments,
    FunctionToolDefinition,
    ParamProperty,
    ParamType,
    ToolCall,
    ToolParamsSchema,
)
from openai.types.chat.chat_completion_audio import ChatCompletionAudio
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
from unsloth import standardize_sharegpt, to_sharegpt

# def _convert_args(args):
#     """
#     Convert a list of arguments in the form ['n=26', 'k=5'] to a string and a dictionary.

#     Parameters:
#     args (list): List of arguments in the form ['n=26', 'k=5']

#     Returns:
#     tuple: (arg_string, arg_dict)
#         arg_string (str): String representation of the arguments, e.g. "n=26, k=5"
#         arg_dict (dict): Dictionary representation of the arguments, e.g. {"n": 26, "k": 5}
#     """
#     arg_dict = {}
#     arg_parts = []

#     print('args:')
#     print(args)

#     for arg in args:
#         print('arg:')
#         print(arg)
#         key, value = arg.split('=')
#         # arg_dict[key] = int(value)
#         arg_dict[key] = value
#         arg_parts.append(f"{key}={value}")

#     arg_string = ", ".join(arg_parts)

#     return arg_string, arg_dict


# def _extract_function_calls(call_list):
#     # This regex captures the function name and the arguments
#     pattern = r"(\w+)\((.*)\)"

#     extracted_calls = []

#     for call in call_list:
#         match = re.match(pattern, call)

#         if match:
#             func_name = match.group(1)
#             args = match.group(2)
#             # Split the arguments by commas, handling nested tuples
#             arg_list = []
#             nested_level = 0
#             current_arg = []

#             for char in args:
#                 if char == "(":
#                     nested_level += 1

#                 elif char == ")":
#                     nested_level -= 1

#                 if char == "," and nested_level == 0:
#                     arg_list.append("".join(current_arg).strip())
#                     current_arg = []

#                 else:
#                     current_arg.append(char)

#             # Add the last argument if exists
#             if current_arg:
#                 arg_list.append("".join(current_arg).strip())

#             args = _convert_args(arg_list)

#             # extracted_calls.append((func_name, arg_list))
#             extracted_calls.append((func_name, args))

#     # Verification step
#     for func_name, args in extracted_calls:
#         arg_string, arg_dict = args

#         # reconstructed_call = f"{func_name}({', '.join(args)})"
#         reconstructed_call = f"{func_name}({arg_string})"

#         print('\ncall_list:')
#         print(call_list)

#         print('func_name:')
#         print(func_name)

#         print('args:')
#         print(args)

#         print('reconstructed_call:')
#         print(reconstructed_call)

#         if reconstructed_call not in call_list:
#             print(
#                 f"Warning: Cannot recreate original string for {func_name} with args {args}"
#             )
#             raise Exception("Extraction failed")

#     return extracted_calls


@dataclass
class Message:
    audio: Optional[ChatCompletionAudio] = None
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
    refusal: Optional[str] = None
    role: Literal["assistant", "system", "user", "tool"] = "user"
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

    def to_dict(self):
        return {
            "audio": self.audio.to_dict() if self.audio else None,
            "content": self.content,
            "name": self.name,
            "refusal": self.refusal,
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "tool_calls": (
                [tool_call.to_dict() for tool_call in self.tool_calls]
                if self.tool_calls
                else None
            ),
        }


def _parse_chat_threads(
    gorilla_berkeley_function_call_leaderboard_v3_df: pd.DataFrame,
) -> pd.DataFrame:
    chat_threads = []

    for index, row in gorilla_berkeley_function_call_leaderboard_v3_df.iterrows():
        try:
            execution_result_type = row["execution_result_type"]

            # If any of the execution results are not exact matches, skip the example
            if any([x != "exact_match" for x in execution_result_type]):
                # print("execution_result_type")
                # print(execution_result_type)

                # print("Skipping...") # TODO: address these cases
                continue

            question = row["question"]
            assert len(question) == 1, f"{len(question)} != 1"

            ground_truth = row["ground_truth"]

            # print('\nfunction:')
            # print(row["function"])

            tools = []

            for function in row["function"]:
                # print('\nfunction:')
                # print(function)

                properties: Dict[str, ParamProperty] = {}

                for prop_name, prop in function["parameters"]["properties"].items():
                    # print(f"\n{prop_name}:")
                    # print(prop)

                    prop_type = prop["type"]

                    if prop_type == "dict":
                        prop_type = "object"

                    elif prop_type == "float":
                        prop_type = "number"

                    elif prop_type == "tuple":
                        prop_type = "array"

                    assert prop_type in [
                        "string",
                        "number",
                        "integer",
                        "object",
                        "array",
                        "boolean",
                        "null",
                    ], f"{prop_type} not in ['string', 'number', 'integer', 'object', 'array', 'boolean', 'null']"

                    enum: List[str] | None = prop.get("enum")

                    if prop_type == "array":
                        items_type = prop["items"]["type"]

                        if items_type == "float":
                            items_type = "number"

                        elif items_type == "tuple":
                            items_type = "array"

                        assert items_type in [
                            "string",
                            "number",
                            "integer",
                            "object",
                            "array",
                            "boolean",
                            "null",
                        ], f"{items_type} not in ['string', 'number', 'integer', 'object', 'array', 'boolean', 'null']"

                        items = ParamType(
                            type=items_type,
                        )

                    else:
                        items = None

                    properties[prop_name] = ParamProperty(
                        description=prop.get("description"),
                        enum=enum,
                        items=items,
                        type=prop_type,
                    )

                tool_params_schema_type = function["parameters"]["type"]

                if tool_params_schema_type == "dict":
                    tool_params_schema_type = "object"

                assert (
                    tool_params_schema_type == "object"
                ), f"{tool_params_schema_type} != object"

                parameters = ToolParamsSchema(
                    additionalProperties=function["parameters"].get(
                        "additionalProperties"
                    ),
                    properties=properties,
                    required=function["parameters"].get("required"),
                    type=tool_params_schema_type,
                )

                tool = FunctionToolDefinition(
                    description=function["description"],
                    name=function["name"],
                    parameters=parameters,
                    strict=function.get("strict", False),
                )

                tools.append(tool)

            # print('\ntools:')
            # print(tools)

            # print('\nground_truth:')
            # print(ground_truth)

            tool_calls = []

            for function_call_signature in ground_truth:
                # print('\nfunction_call_signature:')
                # print(function_call_signature) # e.g. calc_binomial_probability(n=20, k=5, p=1/6)

                function_call_name = function_call_signature.split("(")[
                    0
                ]  # e.g. 'calc_binomial_probability'

                # print('\nfunction_call_name:')
                # print(function_call_name)

                # function_call_arguments = function_call_signature.split("(")[1][:-1] # strip off the trailing ')', e.g. 'n=20, k=5, p=1/6'
                function_call_arguments = (
                    function_call_signature.replace(function_call_name, "")
                    .strip("(")
                    .strip(")")
                )  # strip off the leading and trailing '(', ')', e.g. 'n=20, k=5, p=1/6'

                # print('\nfunction_call_arguments:')
                # print(function_call_arguments)

                function_call_arguments_json = {}

                # print('function_call_name:')
                # print(function_call_name)

                tool = next((x for x in tools if x.name == function_call_name), None)

                # print('tool:')
                # print(tool)

                tool_parameters = tool.parameters

                # print('\ntool_parameters:')
                # print(tool_parameters)

                for prop_name, prop in tool_parameters.properties.items():
                    # print(f"\n{prop_name}:")
                    # print(prop)

                    if f"{prop_name}=" in function_call_arguments:
                        if prop.type == "array":
                            # prop_value = re.findall(r"\[(.*?)\]", function_call_arguments.split(f"{prop_name}=")[1].split(",")[0])
                            # prop_value = function_call_arguments.split(f"{prop_name}=")[1]
                            # print('\nfunction_call_arguments:')
                            # print(function_call_arguments)
                            # splits = function_call_arguments.split("=")
                            # print('\splits:')
                            # print(splits)
                            # prop_value_position = splits.index(f"{prop_name}") + 1
                            # prop_value = splits[prop_value_position]
                            # last_equals_position = function_call_arguments.rfind("=")
                            prop_value = function_call_arguments.split(f"{prop_name}=")[
                                1
                            ].split("=")[0]
                            # print('\nprop_value (before):')
                            # print(prop_value)

                            index_of_opening_bracket_or_paren = (
                                prop_value.find("[")
                                if "[" in prop_value
                                else prop_value.find("(")
                            )
                            index_of_closing_bracket_or_paren = (
                                prop_value.rfind("]")
                                if "]" in prop_value
                                else prop_value.rfind(")")
                            )
                            # prop_value = prop_value[: index_of_last_end_bracket + 1]

                            if index_of_closing_bracket_or_paren != -1:
                                prop_value = prop_value[
                                    : index_of_closing_bracket_or_paren + 1
                                ]

                            else:
                                assert (
                                    index_of_opening_bracket_or_paren != -1
                                ), f"{index_of_opening_bracket_or_paren} == -1"

                                if prop_value[0] == "[":
                                    prop_value = prop_value + "]"

                                elif prop_value[0] == "(":
                                    prop_value = prop_value + ")"

                                else:
                                    raise Exception(
                                        f"Unsupported prop_value: {prop_value}"
                                    )

                            # print('\nprop_value (after):')
                            # print(prop_value)

                        elif prop.type == "boolean":
                            prop_value = eval(
                                function_call_arguments.split(f"{prop_name}=")[1].split(
                                    ","
                                )[0]
                            )

                        elif prop.type == "integer":
                            prop_value = int(
                                function_call_arguments.split(f"{prop_name}=")[1].split(
                                    ","
                                )[0]
                            )

                        elif prop.type == "number":
                            # prop_value = eval(
                            #     function_call_arguments.split(f"{prop_name}=")[1].split(
                            #         ","
                            #     )[0]
                            # )
                            prop_value = function_call_arguments.split(f"{prop_name}=")[
                                1
                            ].split(",")[0]

                        elif prop.type == "object":
                            prop_value = eval(
                                function_call_arguments.split(f"{prop_name}=")[1].split(
                                    ","
                                )[0]
                            )

                        elif prop.type == "string":
                            prop_value = function_call_arguments.split(f"{prop_name}=")[
                                1
                            ].split(",")[0]

                        else:
                            raise Exception(f"Unsupported prop type: {prop.type}")

                        function_call_arguments_json[prop_name] = prop_value

                    elif len(tool_parameters.properties) == 1:
                        # If there is only one parameter, assume it is the only one
                        if prop.type == "integer":
                            prop_value = int(function_call_arguments)

                        else:
                            prop_value = eval(function_call_arguments)

                        function_call_arguments_json[prop_name] = prop_value

                # print('\nfunction_call_arguments_json:')
                # print(function_call_arguments_json)

                # print('\nrebuilt_function_call_signature:')
                # print(rebuilt_function_call_signature)

                for prop_name in tool_parameters.required:
                    if prop_name not in function_call_arguments_json:
                        raise Exception(f"Missing required prop: {prop_name}")

                try:
                    try:
                        rebuilt_function_call_signature = f"{function_call_name}({', '.join([f'{k}={v}' for k, v in function_call_arguments_json.items()])})"
                        assert (
                            rebuilt_function_call_signature == function_call_signature
                        ), f"{rebuilt_function_call_signature}\n!=\n{function_call_signature}"

                    except AssertionError as e:
                        rebuilt_function_call_signature = f"{function_call_name}({','.join([f'{k}={v}' for k, v in function_call_arguments_json.items()])})"
                        assert (
                            rebuilt_function_call_signature == function_call_signature
                        ), f"{rebuilt_function_call_signature}\n!=\n{function_call_signature}"

                except AssertionError as e:
                    # print(e)

                    # print("Skipping...")
                    # continue
                    raise e

                # for prop_name in tool_parameters.required:
                #     if prop_name not in function_call_arguments_json:
                #         raise Exception(f"Missing required prop: {prop_name}")

                tool_call = ToolCall(
                    function=FunctionToolCallArguments(
                        arguments=json.dumps(function_call_arguments_json),
                        name=function_call_name,
                    ),
                    id=f"call_{len(tool_calls)}",
                    type="function",
                )

                tool_calls.append(tool_call)

        except AssertionError as e:
            print(e)

            # raise e
            print("Skipping...")
            continue

        # print('\ntool_calls:')
        # print(tool_calls)

        # print('\ncolumns:')
        # print(gorilla_berkeley_function_call_leaderboard_v3_df.columns)

        # print('\nquestion:')
        # print(question)

        messages = [Message(**message) for message in question[0]] + [
            Message(
                role="assistant",
                tool_calls=tool_calls,
            )
        ]

        messages = [message.to_dict() for message in messages]

        # print('\nmessages:')
        # print(messages)

        tools = [tool.to_dict() for tool in tools]

        tools = [
            {
                "function": tool,
                "type": "function",
            }
            for tool in tools
        ]

        # print('\ntools:')
        # print(tools)

        chat_threads.append(
            {
                "documents": [],
                "messages": messages,
                "tools": tools,
            }
        )

    return pd.DataFrame(chat_threads)


def preprocess_chat_threads_df(chat_threads_df: pd.DataFrame) -> pd.DataFrame:
    for _, row in chat_threads_df.iterrows():
        tool_call_old_id_to_tool_call_new_id = {}
        tool_calls = None

        for message in row["messages"]:
            if message["role"] == "assistant" and message.get("tool_calls"):
                tool_calls = message["tool_calls"]

                for i, tool_call in enumerate(tool_calls):
                    tool_call_old_id_to_tool_call_new_id[tool_call["id"]] = f"call_{i}"
                    tool_call["id"] = tool_call_old_id_to_tool_call_new_id[
                        tool_call["id"]
                    ]

            elif message["role"] == "tool":
                message["tool_call_id"] = tool_call_old_id_to_tool_call_new_id[
                    message["tool_call_id"]
                ]

    return chat_threads_df


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


# TODO: Add llamafactory/glaive_toolcall_en dataset to catalog
# See: https://huggingface.co/datasets/llamafactory/glaive_toolcall_en and https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
# Preprocess using unloth's function
def preprocess_llamafactory_glaive_toolcall_en_df(
    # llamafactory_glaive_toolcall_en_df: pd.DataFrame,
    llamafactory_glaive_toolcall_en_df: DatasetDict,
) -> pd.DataFrame:
    # print("llamafactory_glaive_toolcall_en_df.head():")
    # print(llamafactory_glaive_toolcall_en_df.head())
    # print("llamafactory_glaive_toolcall_en_df.column_names:")
    # print(llamafactory_glaive_toolcall_en_df.column_names)

    train_dataset = llamafactory_glaive_toolcall_en_df["train"]

    # print("train_dataset.column_names:")
    # print(train_dataset.column_names)

    # first_example = train_dataset[0]

    # print("first_example:")
    # print(first_example)

    # standardized_dataset = standardize_sharegpt(
    #     train_dataset,
    #     aliases_for_assistant = ["gpt", "assistant", "output", "function_call"],
    #     aliases_for_system    = ["system", "observation"],
    #     aliases_for_user      = ["user", "human", "input",],
    # )

    # standardized_dataset_first_example = standardized_dataset[0]

    # print("standardized_dataset_first_example:")
    # print(standardized_dataset_first_example)

    chat_threads = []

    for example in train_dataset:
        # print("example:")
        # print(example)

        messages = []
        tools = []

        functions = json.loads(example["tools"])

        # print("functions:")
        # print(functions)

        for function in functions:
            properties: Dict[str, ParamProperty] = {}

            for prop_name, prop in function["parameters"]["properties"].items():
                prop_type = prop["type"]

                # if prop_type == "dict":
                #     prop_type = "object"

                # elif prop_type == "float":
                #     prop_type = "number"

                # elif prop_type == "tuple":
                #     prop_type = "array"

                assert prop_type in [
                    "string",
                    "number",
                    "integer",
                    "object",
                    "array",
                    "boolean",
                    "null",
                ], f"{prop_type} not in ['string', 'number', 'integer', 'object', 'array', 'boolean', 'null']"

                enum: List[str] | None = prop.get("enum")

                if prop_type == "array":
                    items_type = prop["items"]["type"]

                    # if items_type == "float":
                    #     items_type = "number"

                    # elif items_type == "tuple":
                    #     items_type = "array"

                    assert items_type in [
                        "string",
                        "number",
                        "integer",
                        "object",
                        "array",
                        "boolean",
                        "null",
                    ], f"{items_type} not in ['string', 'number', 'integer', 'object', 'array', 'boolean', 'null']"

                    items = ParamType(
                        type=items_type,
                    )

                else:
                    items = None

                properties[prop_name] = ParamProperty(
                    description=prop.get("description"),
                    enum=enum,
                    items=items,
                    type=prop_type,
                )

            tool_params_schema_type = function["parameters"]["type"]

            parameters = ToolParamsSchema(
                # additionalProperties=function["parameters"].get("additionalProperties"),
                properties=properties,
                required=function["parameters"].get("required"),
                type=tool_params_schema_type,
            )

            function_tool_definition = FunctionToolDefinition(
                description=function["description"],
                name=function["name"],
                parameters=parameters,
                strict=function.get("strict", False),
            )

            tools.append(
                {
                    "function": function_tool_definition.to_dict(),
                    "type": "function",
                }
            )

        # print('tools:')
        # print(tools)

        tool_calls = []

        for conversation in example["conversations"]:
            # print('conversation:')
            # print(conversation)

            content = None
            role = None
            tool_call = None
            # tool_calls = []

            if conversation["from"] == "function_call":
                function_tool_call_arguments_json = json.loads(conversation["value"])

                tool_call = ToolCall(
                    function=FunctionToolCallArguments(
                        arguments=json.dumps(
                            function_tool_call_arguments_json["arguments"]
                        ),
                        name=function_tool_call_arguments_json["name"],
                    ),
                    id=f"call_{len(tool_calls)}",
                    type="function",
                )

                # tool_calls.append(tool_call)
                role = "assistant"

            elif conversation["from"] == "gpt":
                content = conversation["value"]
                role = "assistant"

            elif conversation["from"] == "human":
                content = conversation["value"]
                role = "user"

            elif conversation["from"] == "observation":
                content = json.loads(conversation["value"])
                role = "tool"

            else:
                raise Exception(
                    f"Unsupported conversation['from']: {conversation['from']}"
                )

            messages.append(
                Message(
                    content=content,
                    role=role,
                    tool_call_id=tool_calls[0].id if tool_calls else None,
                    # tool_calls=tool_calls if tool_calls else None,
                    tool_calls=[tool_call] if tool_call else None,
                ).to_dict()
            )

            if tool_call:
                tool_calls.append(tool_call)

        # print('messages:')
        # print(messages)

        # print('tools:')
        # print(tools)

        chat_threads.append(
            {
                "documents": [],
                "messages": messages,
                "tools": tools,
            }
        )

        # raise NotImplementedError("Preprocessing function not implemented yet")

    return pd.DataFrame(chat_threads)


# TODO: Add Nexusflow/NexusRaven_API_evaluation dataset to catalog
# See: https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation
# Preprocess using unloth's function
def preprocess_nexusflow_nexusraven_api_evaluation_df(
    nexusflow_nexusraven_api_evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    pass


# TODO: Add NousResearch/hermes-function-calling-v1 dataset to catalog
# See: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
# Preprocess using unloth's function
def preprocess_nousresearch_hermes_function_calling_v1_df(
    nousresearch_hermes_function_calling_v1_df: pd.DataFrame,
) -> pd.DataFrame:
    pass


# TODO: Add Team-ACE/ToolACE dataset to catalog
# See: https://huggingface.co/datasets/Team-ACE/ToolACE
# Preprocess using unloth's function
def preprocess_team_ace_toolace_df(team_ace_toolace_df: pd.DataFrame) -> pd.DataFrame:
    pass


# @dataclass
# class Message:
#     content: Optional[
#         Union[
#             str,
#             List[
#                 ChatCompletionContentPartTextParam
#                 | ChatCompletionContentPartImageParam
#                 | ChatCompletionContentPartInputAudioParam
#             ],
#         ]
#     ] = None
#     name: Optional[str] = None
#     role: Literal["assistant", "system", "user", "tool"] = "user"
#     tool_call_id: Optional[str] = None
#     tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


# @dataclass
# class Tool:
#     function: Optional[dict] = None
#     name: Optional[str] = None
#     type: Literal["function"] = "function"


# def _convert_json_list_to_string(json_list):
#     if not isinstance(json_list, list):
#         json_list = []

#     if json_list:
#         tool_call_initial_id_to_tool_call_id = {}

#         for json_item in json_list:
#             try:
#                 message = Message(**json_item)

#                 if message.tool_calls:
#                     for message_tool_call in message.tool_calls:
#                         tool_call_initial_id_to_tool_call_id[
#                             message_tool_call["id"]
#                         ] = f"call_{len(tool_call_initial_id_to_tool_call_id)}"

#                         message_tool_call["id"] = tool_call_initial_id_to_tool_call_id[
#                             message_tool_call["id"]
#                         ]

#                 if message.role == "tool":
#                     message.tool_call_id = tool_call_initial_id_to_tool_call_id[
#                         message.tool_call_id
#                     ]

#                     json_item["tool_call_id"] = message.tool_call_id

#             except TypeError as e:
#                 tool = Tool(**json_item)

#             except TypeError:
#                 raise Exception("Invalid JSON")

#     return json.dumps(json_list)


# Copied and modified from https://cookbook.openai.com/examples/chat_finetuning_data_prep
def validate_model_input_table_df(model_input_table_df: pd.DataFrame) -> pd.DataFrame:
    # Format error checks
    format_errors = defaultdict(int)

    # for ex in dataset:
    for _, row in model_input_table_df.iterrows():
        ex = row.to_dict()

        # print("ex:")
        # print(ex)

        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            raise Exception(f"Invalid data type: {type(ex)}")
            continue

        # messages = ex.get("messages", None)
        messages = json.loads(ex.get("messages", None))

        if not messages:
            format_errors["missing_messages_list"] += 1
            raise Exception("Missing messages list")
            continue

        for message in messages:
            # print("message:")
            # print(message)

            # if "role" not in message or "content" not in message:
            if "role" not in message:
                format_errors["message_missing_key"] += 1
                raise Exception("Missing message key")

            if any(
                k
                not in (
                    "audio",
                    "content",
                    "name",
                    "refusal",
                    "role",
                    "tool_call_id",
                    "tool_calls",
                    "weight",
                )
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1
                raise Exception(f"Unrecognized message key: {message.keys()}")

            if message.get("role", None) not in ("assistant", "system", "tool", "user"):
                format_errors["unrecognized_role"] += 1
                raise Exception(f"Unrecognized role: {message.get('role', None)}")

            content = message.get("content", None)
            tool_calls = message.get("tool_calls", None)

            # if (not content and not tool_calls) or not isinstance(content, str):
            if not content and not tool_calls:
                format_errors["missing_content"] += 1
                raise Exception("Missing content or tool calls")

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1
            raise Exception("Missing assistant message")

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")

        raise Exception("Errors found")

    # else:
    # print("No errors found")


def create_model_input_table_df(
    chat_threads_df: pd.DataFrame,
    drone_training_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df: pd.DataFrame,
    gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df: pd.DataFrame,
    llamafactory_glaive_toolcall_en_df: pd.DataFrame,
    # nexusflow_nexusraven_api_evaluation_df: pd.DataFrame,
    # nousresearch_hermes_function_calling_v1_df: pd.DataFrame,
    # team_ace_toolace_df: pd.DataFrame,
) -> pd.DataFrame:
    model_input_table = pd.concat(
        [
            chat_threads_df,
            drone_training_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_multiple_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_parallel_multiple_df,
            gorilla_berkeley_function_call_leaderboard_v3_exec_simple_df,
            # llamafactory_glaive_toolcall_en_df,
            # nexusflow_nexusraven_api_evaluation_df,
            # nousresearch_hermes_function_calling_v1_df,
            # team_ace_toolace_df,
        ],
        ignore_index=True,
    )

    # TODO: add parallel_tool_calls column

    def _has_parallel_tool_calls(messages):
        for message in messages:
            if (
                message["role"] == "assistant"
                and message["tool_calls"]
                and len(message["tool_calls"]) > 1
            ):
                return True

        return False

    model_input_table["has_parallel_tool_calls"] = model_input_table["messages"].apply(
        _has_parallel_tool_calls
    )

    model_input_table["documents"] = model_input_table["documents"].apply(
        lambda x: json.dumps(x)
        #     _convert_json_list_to_string
    )
    model_input_table["messages"] = model_input_table["messages"].apply(
        lambda x: json.dumps(x)
        #     _convert_json_list_to_string
    )
    model_input_table["tools"] = model_input_table["tools"].apply(
        lambda x: json.dumps(x)
        #     _convert_json_list_to_string
    )

    # sort columns alphabetically
    model_input_table = model_input_table.reindex(
        sorted(model_input_table.columns), axis=1
    )

    # TODO: use hook with Great Expectations to validate the model_input_table
    validate_model_input_table_df(model_input_table)

    return model_input_table
