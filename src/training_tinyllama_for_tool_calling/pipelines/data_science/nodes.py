import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import evaluate
import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from kedro.config import MissingConfigException, OmegaConfigLoader
from kedro.framework.project import settings
from peft import PeftConfig, PeftModel
from pynvml import *
from pynvml_utils import nvidia_smi
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextStreamer,
)
from transformers.trainer_utils import EvalPrediction
from trl import (
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from unsloth import FastLanguageModel

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _get_hf_token() -> str:
    project_root = (
        settings.PROJECT_ROOT if settings.PROJECT_ROOT is not None else Path.cwd()
    )
    # print("project_root: ", project_root)

    # conf_path = str(Path(<project_root>) / settings.CONF_SOURCE)
    conf_path = str(project_root / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    credentials = conf_loader["credentials"]
    hf_token = credentials["dev_hf"]["hf_token"]

    return hf_token


# def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
#     """Splits data into features and targets training and test sets.

#     Args:
#         data: Data containing features and target.
#         parameters: Parameters defined in parameters/data_science.yml.
#     Returns:
#         Split data.
#     """
#     X = data[parameters["features"]]
#     y = data["price"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
#     )
#     return X_train, X_test, y_train, y_test


# def split_data(data: pd.DataFrame, parameters: dict) -> DatasetDict:
# def split_data(data: pd.DataFrame, parameters: dict) -> None:
def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    # def split_data(data: pd.DataFrame, parameters: Dict, credentials: Dict) -> Tuple:
    # X = data[parameters["features"]]
    # y = data["price"]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    # )
    # return X_train, X_test, y_train, y_test

    dataset = Dataset.from_pandas(data)

    # train_test_ds = dataset.train_test_split(seed=SEED, test_size=0.2)
    train_test_ds = dataset.train_test_split(
        seed=parameters["random_state"], test_size=parameters["test_size"]
    )
    # test_validation_ds = train_test_ds["test"].train_test_split(seed=SEED, test_size=0.5)
    test_validation_ds = train_test_ds["test"].train_test_split(
        seed=parameters["random_state"], test_size=0.5
    )

    chat_threads_ds = DatasetDict(
        {
            "train": train_test_ds["train"],
            "test": test_validation_ds["test"],
            "validation": test_validation_ds["train"],
        }
    )

    # https://docs.kedro.org/projects/kedro-datasets/en/kedro-datasets-5.1.0/_modules/kedro_datasets/huggingface/hugging_face_dataset.html#HFDataset
    # AttributeError: 'HFDataset' object has no attribute '_version_cache'
    # return chat_threads_ds

    # TODO: add references to where the examples in the dataset came from to the Hub README
    # chat_threads_ds.push_to_hub(f"mjschock/chat_threads", token=HF_TOKEN)

    print("parameters: ", parameters)

    hf_token = _get_hf_token()
    chat_threads_ds.push_to_hub(f"mjschock/chat_threads", token=hf_token)

    return (
        chat_threads_ds["train"],
        chat_threads_ds["validation"],
        chat_threads_ds["test"],
    )


# https://github.com/huggingface/trl/blob/31b7820aade7cefa06e3d4160dcff8a602a14850/trl/models/utils.py#L44
@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    # bos_token: str = "<|im_start|>"
    bos_token: str = "<s>"
    # eos_token: str = "<|im_end|>"
    eos_token: str = "</s>"
    # pad_token: str = "<|im_end|>"
    pad_token: str = "</s>"
    # pad_token: str = "<unk>"

    @property
    def chat_template(self):
        # Influenced by:
        # https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use
        # https://github.com/abetlen/llama-cpp-python/blob/7c4aead82d349469bbbe7d8c0f4678825873c039/llama_cpp/llama_chat_format.py#L3387
        # https://github.com/Mozilla-Ocho/llamafile/blob/66a84d8aea2990895fc4f64786406fea64e79197/llama.cpp/server/server.cpp#L480 (need <|im_start|> b/c Mozilla)
        # https://github.com/openai/openai-python/blob/120d225b91a8453e15240a49fb1c6794d8119326/chatml.md
        # https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#prompt
        # https://huggingface.co/blog/unified-tool-use
        return (
            "{%- set system_message_present = messages | selectattr('role', 'equalto', 'system') | list -%}"
            "{%- if not system_message_present -%}"
            '{%- set messages = [{ "content": "You are an AI agent acting as a human assistant.", "role": "system" }] + messages -%}'
            "{%- endif -%}"
            "{%- for message in messages -%}"
            # "<|im_start|>{{ message.role }}{{ '\n' }}"
            "<|{{ message.role }}|>{{ '\n' }}"
            # System message
            "{%- if message.role == 'system' -%}"
            "{{ message.content }}"
            "{%- if tools and tools | length > 0 -%}"
            "{{ '\n\n' }}You are aware of the following tools in your environment:{{ '\n' }}"
            "{\n"
            "  \"tools\": [{{ '\n' }}"
            "{%- for tool in tools -%}"
            "{{ '    ' }}{\n"
            '      "function": {\n'
            '        "description": "{{ tool.function.description }}",{{ \'\n\' }}'
            '        "name": "{{ tool.function.name }}",{{ \'\n\' }}'
            "        \"parameters\": {{ tool.function.parameters | tojson }}{{ '\n' }}"
            # "        \"parameters\": {\n"
            # "{{ '        ' }}}\n"
            "      },{{ '\n' }}"
            '      "type": "{{ tool.type }}"{{ \'\n\' }}'
            "    }{%- if not loop.last -%},{%- endif -%}{{ '\n' }}"
            "{%- endfor -%}"
            "{{ '  ' }}]{{ '\n' }}"
            "}"
            "{{ '\n\n' }}If you would like to suggest one or more tool calls, please respond in the following format:{{ '\n' }}"
            "{\n"
            '  "finish_reason": "tool_calls",{{ \'\n\' }}'
            "  \"tool_calls\": [{{ '\n' }}"
            "{{ '    ' }}{\n"
            '      "arguments": "{\\"parameter_name\\": \\"parameter_value\\"}",{{ \'\n\' }}'
            '      "id": "call_id",{{ \'\n\' }}'
            '      "name": "tool_name"{{ \'\n\' }}'
            "    }{{ '\n' }}"
            "  ]{{ '\n' }}"
            "}"
            "{%- endif -%}"
            # "<|im_end|>{{ '\n' }}"
            "{{ eos_token }}{{ '\n' }}"
            "{%- endif -%}"
            # User message
            "{%- if message.role == 'user' -%}"
            "{{ message.content }}"
            # "<|im_end|>{{ '\n' }}"
            "{{ eos_token }}{{ '\n' }}"
            "{%- endif -%}"
            # Assistant message
            "{%- if message.role == 'assistant' -%}"
            "{% generation %}"
            ## Tool calls (Actions)
            "{%- if message.tool_calls and message.tool_calls | length > 0 -%}"
            "{\n"
            '  "finish_reason": "tool_calls",{{ \'\n\' }}'
            "  \"tool_calls\": [{{ '\n' }}"
            "{%- for tool_call in message.tool_calls -%}"
            "{{ '    ' }}{\n"
            "      \"arguments\": {{ tool_call.function.arguments | tojson }},{{ '\n' }}"
            '      "id": "{{ tool_call.id }}",{{ \'\n\' }}'
            '      "name": "{{ tool_call.function.name }}"{{ \'\n\' }}'
            "    }{%- if not loop.last -%},{%- endif -%}{{ '\n' }}"
            "{%- endfor -%}"
            "{{ '  ' }}]{{ '\n' }}"
            "}"
            "{%- else -%}"
            ## Regular message
            "{{ message.content }}"
            "{%- endif -%}"
            "{% endgeneration %}"
            # "<|im_end|>{{ '\n' }}"
            "{{ eos_token }}{{ '\n' }}"
            "{%- endif -%}"
            ## Tool message (Observations)
            "{%- if message.role == 'tool' -%}"
            "{\n"
            "  \"content\": {{ message.content | tojson }},{{ '\n' }}"
            '  "name": "{{ message.name }}",{{ \'\n\' }}'
            '  "tool_call_id": "{{ message.tool_call_id }}"{{ \'\n\' }}'
            "}"
            # "<|im_end|>{{ '\n' }}"
            "{{ eos_token }}{{ '\n' }}"
            "{%- endif -%}"
            "{%- endfor -%}"
            # "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
            # f"{{{{ '{self.assistant}\n' }}}}"
            "{%- if add_generation_prompt -%}"
            + f"{{{{ '{self.assistant}\n' }}}}"
            + "{%- endif -%}"
        )

    @property
    def user(self):
        # return f"{self.bos_token}user"
        return "<|user|>"

    @property
    def assistant(self):
        # return f"{self.bos_token}assistant"
        return "<|assistant|>"

    @property
    def system(self):
        # return f"{self.bos_token}system"
        return "<|system|>"


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}


# https://github.com/huggingface/trl/blob/31b7820aade7cefa06e3d4160dcff8a602a14850/trl/models/utils.py#L78
def _setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml"]] = "chatml",
    resize_to_multiple_of: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    Args:
        model (`~transformers.PreTrainedModel`): The model to be modified.
        tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
        format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
        resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to. Defaults to None.

    Returns:
        model (`~transformers.PreTrainedModel`): The modified model.
        tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    # check if format available and retrieve
    if format not in FORMAT_MAPPING:
        raise ValueError(
            f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}"
        )

    chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                chat_format.bos_token,
                chat_format.eos_token,
                chat_format.pad_token,
            ]
        }
    )

    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=(
            resize_to_multiple_of if resize_to_multiple_of is not None else None
        ),
    )

    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def prepare_base_model(
    train_ds: Dataset,
    validation_ds: Dataset,
    test_ds: Dataset,
) -> str:
    # ) -> dict:
    # ) -> None:
    with mlflow.start_run(
        log_system_metrics=True,
        nested=True,
    ):
        # MAX_LENGTH = 4096

        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # add_eos_token=True,
            # model_max_length=MAX_LENGTH,
            # padding_side="left", # https://www.mlflow.org/docs/2.12.1/llms/transformers/tutorials/fine-tuning/transformers-peft.html#Padding-the-Training-Dataset
            use_fast=True,
        )

        # # quantization_config = BitsAndBytesConfig(
        # #     # Load the model with 4-bit quantization
        # #     load_in_4bit=True,
        # #     # Use double quantization
        # #     bnb_4bit_use_double_quant=True,
        # #     # Use 4-bit Normal Float for storing the base model weights in GPU memory
        # #     bnb_4bit_quant_type="nf4",
        # #     # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
        # #     bnb_4bit_compute_dtype=torch.bfloat16,
        # # )

        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # quantization_config=quantization_config,
        )

        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     # model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        #     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # unsloth/tinyllama-bnb-4bit",
        #     # max_seq_length = max_seq_length,
        #     max_seq_length=MAX_LENGTH,
        #     # dtype = dtype,
        #     dtype=None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        #     # load_in_4bit = load_in_4bit,
        #     load_in_4bit=True # Use 4bit quantization to reduce memory usage. Can be False.
        # )

        print(f"=== Before setup_chat_format ===")

        print(f"tokenizer.bos_token: {tokenizer.bos_token}")
        print(f"tokenizer.eos_token: {tokenizer.eos_token}")
        print(f"tokenizer.pad_token: {tokenizer.pad_token}")

        # print(f"tokenizer.chat_template: {tokenizer.chat_template}")

        # FastLanguageModel.for_inference(model) # Enable native 2x faster inference

        model, tokenizer = _setup_chat_format(model, tokenizer)

        print(f"=== After setup_chat_format ===")

        print(f"tokenizer.bos_token: {tokenizer.bos_token}")
        print(f"tokenizer.eos_token: {tokenizer.eos_token}")
        print(f"tokenizer.pad_token: {tokenizer.pad_token}")

        # print(f"tokenizer.chat_template: {tokenizer.chat_template}")

        # print(f"vocab_size: {tokenizer.vocab_size}")

        # dataset = load_dataset("mjschock/chat_threads", revision=DATASET_VERSION)

        # test_dataset = dataset["test"]
        # example = test_dataset[10]

        # text = tokenizer.apply_chat_template(
        #     add_generation_prompt=True,
        #     documents=json.loads(example["documents"]),
        #     conversation=json.loads(example["messages"]),
        #     tools=json.loads(example["tools"]),
        #     tokenize=False,
        # )

        # print("prompt:")
        # print(text)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # generation_pipeline = transformers.pipeline(
        #     # task="text-generation",
        #     # "text2text-generation",
        #     "text-generation",
        #     # device=device,
        #     # device_map="auto",
        #     # device_map=get_kbit_device_map(),
        #     model=model,
        #     tokenizer=tokenizer,
        # )

        # Define a simple input example that will be recorded with the model in MLflow, giving
        # users of the model an indication of the expected input format.
        # input_example = ["prompt 1", "prompt 2", "prompt 3"]

        # Define the parameters (and their defaults) for optional overrides at inference time.
        # parameters = {"max_length": 512, "do_sample": True, "temperature": 0.4}
        # parameters = {"max_length": MAX_LENGTH, "do_sample": True, "temperature": 0.0}

        # signature = mlflow.models.infer_signature(
        #     input_example,
        #     mlflow.transformers.generate_signature_output(
        #         generation_pipeline, input_example
        #     ),
        #     parameters,
        # )

        # print("signature:")
        # print(signature)

        # mlflow.set_experiment("Transformers Introduction")

        # with mlflow.start_run(nested=True):
        #     model_info = mlflow.transformers.log_model(
        #         transformers_model=generation_pipeline,
        #         artifact_path="text_generator",
        #         input_example=input_example,
        #         signature=signature,
        #         # Uncomment the following line to save the model in 'reference-only' mode:
        #         save_pretrained=False,
        #     )

        # print("model_info:")
        # print(model_info)

        # Get the ID of the MLflow Run that was automatically created above
        # last_run_id = mlflow.last_active_run().info.run_id

        # with mlflow.start_run(
        #     nested=True,
        #     run_id=last_run_id,
        # ):
        #     # mlflow.log_params(peft_config.to_dict())
        #     model_info = mlflow.transformers.log_model(
        #         # transformers_model={"model": trainer.model, "tokenizer": tokenizer_no_pad},
        #         transformers_model={"model": model, "tokenizer": tokenizer},
        #         # prompt_template=prompt_template,
        #         # signature=signature,
        #         artifact_path="model",  # This is a relative path to save model files within MLflow run
        #         save_pretrained=False, # Save the model in 'reference-only' mode
        #     )

        hf_token = _get_hf_token()

        repo_id = "mjschock/TinyLlama-1.1B-Chat-v1.0"

        model.save_pretrained("data/06_models/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.save_pretrained("data/06_models/TinyLlama-1.1B-Chat-v1.0")

        # # model.push_to_hub(repo_id="mjschock/TinyLlama-1.1B-Chat-v1.0", use_auth_token=True)
        model.push_to_hub(
            # repo_id="mjschock/TinyLlama-1.1B-Chat-v1.0",
            repo_id=repo_id,
            # use_auth_token=True,
            token=hf_token,
            # hf_token=hf_token,
        )
        tokenizer.push_to_hub(
            # repo_id="mjschock/TinyLlama-1.1B-Chat-v1.0", use_auth_token=True
            # repo_id="mjschock/TinyLlama-1.1B-Chat-v1.0",
            repo_id=repo_id,
            # use_auth_token=True,
            token=hf_token,
            # hf_token=hf_token,
        )

        # # model.save_pretrained("mjschock/TinyLlama-1.1B-Chat-v1.0")
        # # tokenizer.save_pretrained("mjschock/TinyLlama-1.1B-Chat-v1.0")

        # # return "mjschock/TinyLlama-1.1B-Chat-v1.0"
        # return repo_id

        # return model_info.artifact_path
        # return {
        #     "model": model,
        #     "tokenizer": tokenizer,
        # }

        # transformers_model = {
        #     "model": model,
        #     "tokenizer": tokenizer,
        # }

        # # https://mlflow.org/docs/latest/python_api/mlflow.transformers.html#mlflow.transformers.log_model
        # model_info = mlflow.transformers.log_model(
        #     # transformers_model={"model": trainer.model, "tokenizer": tokenizer_no_pad},
        #     # transformers_model={"model": model, "tokenizer": tokenizer},
        #     transformers_model=transformers_model,
        #     # prompt_template=prompt_template,
        #     # signature=signature,
        #     # artifact_path="model",  # This is a relative path to save model files within MLflow run
        #     artifact_path="pretrained_model",
        #     # save_pretrained=False, # Save the model in 'reference-only' mode
        # )

        # return repo_id

        # return f"{model_info.artifact_path}/model"
        # return model_info.model_uri

        # pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

        # # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a friendly chatbot who always responds in the style of a pirate",
        #     },
        #     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        # ]
        # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        # print(outputs[0]["generated_text"])

        # messages = [
        #     {"role": "user", "content": "Describe a tall tower in the capital of France."},
        # ]
        # inputs = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize = True,
        #     add_generation_prompt = True, # Must add for generation
        #     return_tensors = "pt",
        # ).to("cuda")

        # from transformers import TextStreamer
        # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        # _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
        #                 use_cache = True, temperature = 1.5, min_p = 0.1)

        # return {
        #     "model": model,
        #     "tokenizer": tokenizer,
        # }

        # return components

        # sample = test_ds[1]

        # signature = infer_signature(
        #     model_input=sample["prompt"],
        #     model_output=sample["answer"],
        #     # Parameters are saved with default values if specified
        #     params={"max_new_tokens": 256, "repetition_penalty": 1.15, "return_full_text": False},
        # )

        prompts = []
        responses = []

        for example in test_ds.select(range(1)):
            # example = test_ds[17]
            print("example: ", example)

            text = tokenizer.apply_chat_template(
                add_generation_prompt=True,
                documents=json.loads(example["documents"]),
                conversation=json.loads(example["messages"])[0:-1],
                tools=json.loads(example["tools"]),
                tokenize=False,
            )

            print("prompt:")
            print(text)

            inputs = tokenizer(text, return_tensors="pt")

            streamer = TextStreamer(tokenizer, skip_prompt=True)

            print("response:")
            token_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=512)

            response = tokenizer.decode(
                token_ids[0],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
            )

            prompts.append(text)
            responses.append(response)

        prompt = prompts[0]
        response = responses[0]

        print("prompt:")
        print(prompt)

        print("response:")
        print(response)

        signature = mlflow.models.infer_signature(
            model_input=prompt,
            model_output=response,
            # Parameters are saved with default values if specified
            # params={"max_new_tokens": 512, "repetition_penalty": 1.15, "return_full_text": False},
        )

        print("signature:")
        print(signature)

        model_info = mlflow.transformers.log_model(
            artifact_path="pretrained_model",
            # prompt_template=
            registered_model_name="TinyLlama-1.1B-Chat-v1.0",
            signature=signature,
            task="text-generation",
            # transformers_model={"model": model, "tokenizer": tokenizer},
            transformers_model="data/06_models/TinyLlama-1.1B-Chat-v1.0",
        )

        # return model_info.artifact_path
        return model_info.model_uri


# # def evaluate_base_model(pretrained_model: str, test_ds: Dataset) -> pd.DataFrame:
# def evaluate_base_model(pretrained_model: str, test_ds: Dataset) -> pd.DataFrame:
#     with mlflow.start_run(
#         log_system_metrics=True,
#         nested=True,
#     ):

#         print("type(pretrained_model):")
#         print(pretrained_model)

#         # tool_model = mlflow.pyfunc.load_model(model_info.model_uri)
#         model = mlflow.transformers.load_model(
#             device="cpu",
#             dst_path="data/06_models/TinyLlama-1.1B-Chat-v1.0",
#             model_uri=pretrained_model,
#             return_type="components",
#         )

#         print("model:")
#         print(model)

#         # model, tokenizer = model["model"], model["tokenizer"]

#         # model = AutoModelForCausalLM.from_pretrained(pretrained_model)
#         # tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#         # model, tokenizer = FastLanguageModel.from_pretrained(
#         #     # model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         #     model_name=pretrained_model,
#         #     # max_seq_length = max_seq_length,
#         #     # dtype = dtype,
#         #     # load_in_4bit = load_in_4bit,
#         # )
#         # FastLanguageModel.for_inference(model) # Enable native 2x faster inference

#         prompts = []
#         responses = []

#         # # for example in test_ds:
#         # # for example in test_ds[0:1]:
#         for example in test_ds.select(range(1)):
#             # example = test_ds[17]
#             print("example: ", example)

#             # text = pretrained_model.tokenizer.apply_chat_template(
#             text = model["tokenizer"].apply_chat_template(
#                 add_generation_prompt=True,
#                 documents=json.loads(example["documents"]),
#                 conversation=json.loads(example["messages"])[0:-1],
#                 tools=json.loads(example["tools"]),
#                 tokenize=False,
#             )

#             print("prompt:")
#             print(text)

#             # inputs = pretrained_model.tokenizer(text, return_tensors="pt")
#             inputs = model["tokenizer"](text, return_tensors="pt")

#             # streamer = TextStreamer(pretrained_model.tokenizer, skip_prompt=True)
#             streamer = TextStreamer(model["tokenizer"], skip_prompt=True)

#             print("response:")
#             # token_ids = pretrained_model.model.generate(
#             token_ids = model["model"].generate(
#                 **inputs, streamer=streamer, max_new_tokens=512
#             )

#             # response = pretrained_model.tokenizer.decode(
#             response = model["tokenizer"].decode(
#                 token_ids[0],
#                 clean_up_tokenization_spaces=True,
#                 skip_special_tokens=False,
#             )

#             prompts.append(text)
#             responses.append(response)

#         # # with open("pretrain_inference_check_response.txt", "w") as f:
#         # #     f.write(response)

#         # # df = pd.DataFrame(
#         # #     {
#         # #         "prompt": [text],
#         # #         "response": [response],
#         # #     }
#         # # )

#         df = pd.DataFrame(
#             {
#                 "prompt": prompts,
#                 "response": responses,
#             }
#         )

#         return df


# def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
# def train_model(chat_threads_ds: DatasetDict) -> LinearRegression:
# def train_model(chat_threads_train_ds: Dataset, chat_threads_validation_ds: Dataset, parameters: Dict) -> LinearRegression:
def train_model(
    chat_threads_train_ds: Dataset,
    chat_threads_validation_ds: Dataset,
    chat_threads_test_ds: Dataset,
    model_config: Dict,
    sft_config: Dict,
    sft_script_arguments: Dict,
    pretrained_model_uri: str,
    # ) -> LinearRegression:
) -> str:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    # print("chat_threads_ds: ", chat_threads_ds)
    # X_train = chat_threads_ds["train"]
    # y_train = chat_threads_ds["validation"]
    # regressor = LinearRegression()
    # regressor.fit(X_train, y_train)
    # return regressor

    # from training_tinyllama_for_tool_calling.pipelines.data_science.tune import main

    # print('parameters: ', parameters)

    print("model_config:")
    print(model_config)

    print("sft_config:")
    print(sft_config)

    print("sft_script_arguments:")
    print(sft_script_arguments)

    # experiment = mlflow.set_experiment("Training TinyLlama for Tool-calling")
    # # Get Experiment Details
    # print(f"Experiment_id: {experiment.experiment_id}")
    # print(f"Artifact Location: {experiment.artifact_location}")
    # print(f"Tags: {experiment.tags}")
    # print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

    model_config = ModelConfig(**model_config)
    sft_config = SFTConfig(**sft_config, hub_token=_get_hf_token())
    sft_script_arguments = SFTScriptArguments(**sft_script_arguments)

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
        use_cache=False if sft_config.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    # sft_config.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, revision=DATASET_VERSION)
    # dataset = load_dataset(sft_script_arguments.dataset_name, revision=DATASET_VERSION)
    dataset = load_dataset(sft_script_arguments.dataset_name)

    # eval_dataset = dataset[script_args.dataset_test_split]
    # eval_dataset = dataset[sft_script_arguments.dataset_test_split]
    # test_dataset = dataset[script_args.dataset_test_split]
    # test_dataset = dataset[sft_script_arguments.dataset_test_split]
    # train_dataset = dataset[script_args.dataset_train_split]
    train_dataset = dataset[sft_script_arguments.dataset_train_split]
    validation_dataset = dataset[sft_script_arguments.dataset_test_split]

    def load_and_preprocess_data(dataset, tokenizer):
        """
        Load and preprocess the dataset for training.

        Args:
            dataset: The dataset to preprocess
            tokenizer: Tokenizer to use for preprocessing.

        Returns:
            datasets.Dataset: Preprocessed dataset.
        """

        def preprocess_function(examples):
            # Extract the messages from the example
            conversation = examples["messages"]
            documents = examples.get("documents", [])
            tools = examples.get("tools", [])

            # Apply chat template to generate tokenized input and assistant mask
            tokenized_output = tokenizer.apply_chat_template(
                add_generation_prompt=False,
                conversation=json.loads(conversation),
                documents=json.loads(documents),
                max_length=4096,
                padding="longest",
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                tokenize=True,
                tools=json.loads(tools),
                truncation=True,  # TODO: verify we're not truncating anything in the datasets
            )

            # Extract the input IDs and assistant tokens mask
            input_ids = tokenized_output["input_ids"][0]
            assistant_masks = torch.tensor(tokenized_output["assistant_masks"])
            attention_mask = tokenized_output["attention_mask"][0]

            # Use the assistant mask to create labels
            labels = torch.where(assistant_masks == 1, input_ids, torch.tensor(-100))

            return {
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "labels": labels,
            }

        # Preprocess the dataset
        return dataset.map(
            preprocess_function,
            batched=False,
            num_proc=1,
            remove_columns=dataset.column_names,
        )  # TODO: use batched=True

    # tokenized_test_dataset = load_and_preprocess_data(test_dataset, tokenizer)
    tokenized_train_dataset = load_and_preprocess_data(
        train_dataset, tokenizer
    )  # TODO: just do train_dataset.map(tokenize_function)
    tokenized_validation_dataset = load_and_preprocess_data(
        validation_dataset, tokenizer
    )

    ################
    # Training
    ################
    print("model:")
    print(model)

    print("model_config.lora_target_modules:", model_config.lora_target_modules)

    peft_config = get_peft_config(model_config)

    # metrics = evaluate.combine(["bertscore", "bleu", "meteor", "rouge"]) # TODO: need to add 'lang': 'en' or 'model_type' for bertscore
    metrics = evaluate.combine(["bleu", "meteor", "rouge"])
    metrics_tracker = {}

    def compute_metrics(
        eval_pred: EvalPrediction, compute_result: bool
    ) -> (
        Dict
    ):  # metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        # compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
        #     The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
        #     a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
        #     `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
        #     after the last eval batch to signal that the function needs to calculate and return the global summary
        #     statistics rather than accumulating the batch-level statistics.
        print("=== compute_metrics ===")

        assert isinstance(
            eval_pred, EvalPrediction
        ), f"Expected EvalPrediction, got {type(eval_pred)}"

        all_labels = eval_pred.label_ids
        all_preds = eval_pred.predictions
        is_last_step = compute_result

        all_labels[all_labels == -100] = tokenizer.pad_token_id
        references: List[str] = tokenizer.batch_decode(
            all_labels, skip_special_tokens=True
        )

        assert (
            all_preds.shape == all_labels.shape
        ), f"Expected predictions and labels to have the same shape, got {all_preds.shape} and {all_labels.shape}"

        predictions: List[str] = tokenizer.batch_decode(
            all_preds, skip_special_tokens=True
        )

        assert len(predictions) == len(
            references
        ), f"Expected predictions and references to have the same length, got {len(predictions)} and {len(references)}"

        eval_batch_metrics = metrics.compute(
            predictions=predictions,
            references=references,
            # use_aggregator=True,  # Use the aggregator to get the average of the metrics
        )

        computed_metrics = {}

        print("eval_batch_metrics:")
        print(eval_batch_metrics)

        for key, value in eval_batch_metrics.items():
            if type(value) in [list, np.ndarray]:
                value = np.mean(value)

            metrics_tracker[key] = np.mean([metrics_tracker.get(key, 0.0), value])
            computed_metrics[key] = metrics_tracker[key]

            if is_last_step:
                metrics_tracker[key] = 0.0

        print("computed_metrics:")
        print(computed_metrics)

        return computed_metrics

    def preprocess_logits_for_metrics(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)

        return pred_ids

    trainer = SFTTrainer(
        # args=training_args,
        args=sft_config,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer),
        # eval_dataset=tokenized_eval_dataset,
        eval_dataset=tokenized_validation_dataset,
        model=model,
        peft_config=peft_config,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
    )

    print("num_trainable_parameters:")
    print(trainer.get_num_trainable_parameters())

    mlflow.autolog()

    # Start MLflow run
    with mlflow.start_run(
        log_system_metrics=True,
        nested=True,
        # run_name=training_args.output_dir,
        # run_name=sft_config.output_dir,
    ) as run:
        # try:
        #     hf_token = _get_hf_token()

        #     train_result = trainer.train(
        #         resume_from_checkpoint=True,
        #     )

        # except ValueError as e:
        #     if "checkpoint" in str(e):
        #         train_result = trainer.train()

        #     else:
        #         raise e

        train_result = trainer.train()

        print("train_result:")
        print(train_result)

        # Save and push to hub
        # trainer.save_model(training_args.output_dir)
        trainer.save_model(sft_config.output_dir)

        # if training_args.push_to_hub:
        if sft_config.push_to_hub:
            # trainer.push_to_hub(dataset=[script_args.dataset_name])
            trainer.push_to_hub(dataset=[sft_script_arguments.dataset_name])

        # try:
        #     predictions = trainer.predict(tokenized_test_dataset)

        # except torch.OutOfMemoryError as e:
        #     print("Out of memory error in predictions. Trying with one example.")

        #     test_dataset_of_one = tokenized_test_dataset.select([0])
        #     predictions = trainer.predict(test_dataset_of_one)

        # print("predictions:")
        # print(predictions)

        # train_result = trainer.train()

        # print("mlflow.MlflowClient().get_run(run.info.run_id).data:")
        # print(mlflow.MlflowClient().get_run(run.info.run_id).data)

        # return f"{pretrained_model}-sft-chat_threads"

        # components = {"model": trainer.model, "tokenizer": trainer.tokenizer}

        # model_info = mlflow.transformers.log_model(
        #     # transformers_model={"model": trainer.model, "tokenizer": tokenizer_no_pad},
        #     # transformers_model={"model": trainer.model, "tokenizer": trainer.tokenizer}, # TODO: use tokenizer w/ left padding during training?
        #     transformers_model=components,
        #     # prompt_template=prompt_template,
        #     # signature=signature,
        #     # artifact_path="model",  # This is a relative path to save model files within MLflow run
        #     artifact_path="tuned_model",
        #     # save_pretrained=False, # Save the model in 'reference-only' mode
        # )

        # return repo_id

        # return model_info.artifact_path

        # return {
        #     "model": model,
        #     "tokenizer": tokenizer,
        # }

        # return components

        prompts = []
        responses = []

        # for example in test_ds.select(range(1)):
        # for example in chat_threads_validation_ds.select(range(1)):
        for example in chat_threads_test_ds.select(range(1)):
            # example = test_ds[17]
            print("example: ", example)

            text = tokenizer.apply_chat_template(
                add_generation_prompt=True,
                documents=json.loads(example["documents"]),
                conversation=json.loads(example["messages"])[0:-1],
                tools=json.loads(example["tools"]),
                tokenize=False,
            )

            print("prompt:")
            print(text)

            inputs = tokenizer(text, return_tensors="pt")

            streamer = TextStreamer(tokenizer, skip_prompt=True)

            print("response:")
            token_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=512)

            response = tokenizer.decode(
                token_ids[0],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
            )

            prompts.append(text)
            responses.append(response)

        prompt = prompts[0]
        response = responses[0]

        print("prompt:")
        print(prompt)

        print("response:")
        print(response)

        signature = mlflow.models.infer_signature(
            model_input=prompt,
            model_output=response,
            # Parameters are saved with default values if specified
            # params={"max_new_tokens": 512, "repetition_penalty": 1.15, "return_full_text": False},
        )

        print("signature:")
        print(signature)

        model_info = mlflow.transformers.log_model(
            # artifact_path="pretrained_model",
            artifact_path="tuned_model",
            # prompt_template=
            registered_model_name="TinyLlama-1.1B-Chat-v1.0-sft-chat_threads",
            signature=signature,
            task="text-generation",
            # transformers_model={"model": model, "tokenizer": tokenizer},
            transformers_model={"model": trainer.model, "tokenizer": trainer.tokenizer},
            # transformers_model="data/06_models/TinyLlama-1.1B-Chat-v1.0-sft-chat_threads",
        )

        # return model_info.artifact_path
        return model_info.model_uri


# def evaluate_model(
#     # regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
#     regressor: LinearRegression, chat_threads_ds: DatasetDict
# ):
#     """Calculates and logs the coefficient of determination.

#     Args:
#         regressor: Trained model.
#         X_test: Testing data of independent features.
#         y_test: Testing data for price.
#     """
#     y_pred = regressor.predict(X_test)
#     score = r2_score(y_test, y_pred)
#     logger = logging.getLogger(__name__)
#     logger.info("Model has a coefficient R^2 of %.3f on test data.", score)


# def evaluate_model(
#     # pretrained_model: str, tuned_model: str, test_ds: Dataset
#     pipeline,
#     test_ds: Dataset,
# ) -> pd.DataFrame:
#     # model = AutoModelForCausalLM.from_pretrained(pretrained_model)
#     # tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#     # model = AutoModelForCausalLM.from_pretrained("mjschock/TinyLlama-1.1B-Chat-v1.0")
#     # tokenizer = AutoTokenizer.from_pretrained("mjschock/TinyLlama-1.1B-Chat-v1.0")
#     # config = PeftConfig.from_pretrained(
#     #     "mjschock/TinyLlama-1.1B-Chat-v1.0-sft-chat_threads"
#     # )
#     # base_model = AutoModelForCausalLM.from_pretrained("mjschock/TinyLlama-1.1B-Chat-v1.0")
#     # base_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
#     # model = PeftModel.from_pretrained(
#     #     # base_model, "mjschock/TinyLlama-1.1B-Chat-v1.0-sft-chat_threads"
#     #     base_model,
#     #     tuned_model,
#     # )
#     # # tokenizer = AutoTokenizer.from_pretrained("mjschock/TinyLlama-1.1B-Chat-v1.0")
#     # tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

#     # dataset = load_dataset("mjschock/chat_threads", revision=DATASET_VERSION)

#     # test_dataset = dataset["test"]
#     # example = test_dataset[17]

#     prompts = []
#     responses = []

#     # for example in test_ds:
#     # for example in test_ds[0:1]:
#     for example in test_ds.select(range(1)):
#         # example = test_ds[17]
#         print("example: ", example)

#         text = pipeline.tokenizer.apply_chat_template(
#             add_generation_prompt=True,
#             documents=json.loads(example["documents"]),
#             conversation=json.loads(example["messages"])[0:-1],
#             tools=json.loads(example["tools"]),
#             tokenize=False,
#         )

#         print("prompt:")
#         print(text)

#         inputs = pipeline.tokenizer(text, return_tensors="pt")

#         streamer = TextStreamer(pipeline.tokenizer, skip_prompt=True)

#         print("response:")
#         token_ids = pipeline.model.generate(
#             **inputs, streamer=streamer, max_new_tokens=512
#         )

#         response = pipeline.tokenizer.decode(
#             token_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=False
#         )

#         prompts.append(text)
#         responses.append(response)

#     # with open("pretrain_inference_check_response.txt", "w") as f:
#     #     f.write(response)

#     # df = pd.DataFrame(
#     #     {
#     #         "prompt": [text],
#     #         "response": [response],
#     #     }
#     # )

#     df = pd.DataFrame(
#         {
#             "prompt": prompts,
#             "response": responses,
#         }
#     )

#     return df


# def evaluate_base_model(pretrained_model: str, test_ds: Dataset) -> pd.DataFrame:
def evaluate_model(model_uri: str, test_ds: Dataset) -> pd.DataFrame:
    with mlflow.start_run(
        log_system_metrics=True,
        nested=True,
    ):

        # print("type(pretrained_model):")
        # print(pretrained_model)

        # print("model_uri:")
        # print(model_uri)

        # raise Exception("model_uri: ", model_uri)

        # tool_model = mlflow.pyfunc.load_model(model_info.model_uri)
        model = mlflow.transformers.load_model(
            device="cpu",
            # dst_path="data/06_models/TinyLlama-1.1B-Chat-v1.0",
            model_uri=model_uri,
            return_type="components",
        )

        print("model:")
        print(model)

        # model, tokenizer = model["model"], model["tokenizer"]

        # model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        # tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     # model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        #     model_name=pretrained_model,
        #     # max_seq_length = max_seq_length,
        #     # dtype = dtype,
        #     # load_in_4bit = load_in_4bit,
        # )
        # FastLanguageModel.for_inference(model) # Enable native 2x faster inference

        prompts = []
        responses = []

        # # for example in test_ds:
        # # for example in test_ds[0:1]:
        for example in test_ds.select(range(1)):
            # example = test_ds[17]
            print("example: ", example)

            # text = pretrained_model.tokenizer.apply_chat_template(
            text = model["tokenizer"].apply_chat_template(
                add_generation_prompt=True,
                documents=json.loads(example["documents"]),
                conversation=json.loads(example["messages"])[0:-1],
                tools=json.loads(example["tools"]),
                tokenize=False,
            )

            print("prompt:")
            print(text)

            # inputs = pretrained_model.tokenizer(text, return_tensors="pt")
            inputs = model["tokenizer"](text, return_tensors="pt")

            # streamer = TextStreamer(pretrained_model.tokenizer, skip_prompt=True)
            streamer = TextStreamer(model["tokenizer"], skip_prompt=True)

            print("response:")
            # token_ids = pretrained_model.model.generate(
            token_ids = model["model"].generate(
                **inputs, streamer=streamer, max_new_tokens=512
            )

            # response = pretrained_model.tokenizer.decode(
            response = model["tokenizer"].decode(
                token_ids[0],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
            )

            prompts.append(text)
            responses.append(response)

        # # with open("pretrain_inference_check_response.txt", "w") as f:
        # #     f.write(response)

        # # df = pd.DataFrame(
        # #     {
        # #         "prompt": [text],
        # #         "response": [response],
        # #     }
        # # )

        df = pd.DataFrame(
            {
                "prompt": prompts,
                "response": responses,
            }
        )

        return df
