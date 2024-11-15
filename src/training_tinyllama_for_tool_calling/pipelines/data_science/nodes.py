import json
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple

import evaluate
import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from pynvml import *
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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

    conf_path = str(project_root / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    credentials = conf_loader["credentials"]
    hf_token = credentials["dev_hf"]["hf_token"]

    return hf_token


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    dataset = Dataset.from_pandas(data)

    # train_test_ds = dataset.train_test_split(
    #     seed=parameters["random_state"], test_size=parameters["test_size"]
    # )
    # test_validation_ds = train_test_ds["test"].train_test_split(
    #     seed=parameters["random_state"], test_size=0.5
    # )

    # build test_set from single example using the first example in the dataset
    test_ds = dataset.select(range(1))

    # build train and validation sets from the rest of the dataset
    train_validation_ds = dataset.select(range(1, len(dataset))).train_test_split(
        seed=parameters["random_state"], test_size=parameters["test_size"]
    )

    chat_threads_ds = DatasetDict(
        {
            # "train": train_test_ds["train"],
            "train": train_validation_ds["train"],
            # "test": test_validation_ds["test"],
            "test": test_ds,
            # "validation": test_validation_ds["train"],
            "validation": train_validation_ds["test"],
        }
    )

    # https://docs.kedro.org/projects/kedro-datasets/en/kedro-datasets-5.1.0/_modules/kedro_datasets/huggingface/hugging_face_dataset.html#HFDataset
    # AttributeError: 'HFDataset' object has no attribute '_version_cache'
    # return chat_threads_ds

    # TODO: add references to where the examples in the dataset came from to the Hub README
    # chat_threads_ds.push_to_hub(f"mjschock/chat_threads", token=HF_TOKEN)

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
    with mlflow.start_run(
        log_system_metrics=True,
        nested=True,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            use_fast=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )

        print(f"tokenizer.bos_token: {tokenizer.bos_token}")
        print(f"tokenizer.eos_token: {tokenizer.eos_token}")
        print(f"tokenizer.pad_token: {tokenizer.pad_token}")

        model, tokenizer = _setup_chat_format(model, tokenizer)

        print(f"=== After setup_chat_format ===")

        print(f"tokenizer.bos_token: {tokenizer.bos_token}")
        print(f"tokenizer.eos_token: {tokenizer.eos_token}")
        print(f"tokenizer.pad_token: {tokenizer.pad_token}")

        hf_token = _get_hf_token()

        repo_id = "mjschock/TinyLlama-1.1B-Chat-v1.0"

        model.save_pretrained("data/06_models/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.save_pretrained("data/06_models/TinyLlama-1.1B-Chat-v1.0")

        model.push_to_hub(
            repo_id=repo_id,
            token=hf_token,
        )
        tokenizer.push_to_hub(
            repo_id=repo_id,
            token=hf_token,
        )

        prompts = []
        responses = []

        for example in test_ds.select(range(1)):
            text = tokenizer.apply_chat_template(
                add_generation_prompt=True,
                documents=json.loads(example["documents"]),
                conversation=json.loads(example["messages"])[0:-1],
                tools=json.loads(example["tools"]),
                tokenize=False,
            )

            inputs = tokenizer(text, return_tensors="pt")

            streamer = TextStreamer(tokenizer, skip_prompt=True)

            token_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=512)

            response = tokenizer.decode(
                token_ids[0][len(inputs["input_ids"][0]) :],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
            )

            prompts.append(text)
            responses.append(response)

        prompt = prompts[0]
        response = responses[0]

        signature = mlflow.models.infer_signature(
            model_input=prompt,
            model_output=response,
        )

        model_info = mlflow.transformers.log_model(
            artifact_path="pretrained_model",
            registered_model_name="TinyLlama-1.1B-Chat-v1.0",
            signature=signature,
            task="text-generation",
            # transformers_model={"model": model, "tokenizer": tokenizer},
            transformers_model="data/06_models/TinyLlama-1.1B-Chat-v1.0",
        )

        return model_info.model_uri


def train_model(
    chat_threads_train_ds: Dataset,
    chat_threads_validation_ds: Dataset,
    chat_threads_test_ds: Dataset,
    pretrained_model_uri: str,
    model_config: Dict,
    sft_config: Dict,
    sft_script_arguments: Dict,
) -> str:
    """Trains the model.

    Args:
        chat_threads_train_ds: Training data.
        chat_threads_validation_ds: Validation data.
        chat_threads_test_ds: Test data.
        model_config: Model configuration.
        sft_config: SFT configuration.
        sft_script_arguments: SFT script arguments.
        pretrained_model_uri: URI of the pretrained model.

    Returns:
        URI of the trained model.
    """
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
        use_cache=False if sft_config.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

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

    tokenized_train_dataset = load_and_preprocess_data(
        chat_threads_train_ds, tokenizer
    )  # TODO: just do chat_threads_train_ds.map(tokenize_function), etc.
    tokenized_validation_dataset = load_and_preprocess_data(
        chat_threads_validation_ds, tokenizer
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

    def compute_metrics(eval_pred: EvalPrediction, compute_result: bool) -> Dict:
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
        )

        computed_metrics = {}

        for key, value in eval_batch_metrics.items():
            if type(value) in [list, np.ndarray]:
                value = np.mean(value)

            metrics_tracker[key] = np.mean([metrics_tracker.get(key, 0.0), value])
            computed_metrics[key] = metrics_tracker[key]

            if is_last_step:
                metrics_tracker[key] = 0.0

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
        args=sft_config,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer),
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
    ) as run:
        train_result = trainer.train()

        print("train_result:")
        print(train_result)

        # Save and push to hub
        trainer.save_model(sft_config.output_dir)

        if sft_config.push_to_hub:
            trainer.push_to_hub(dataset=[sft_script_arguments.dataset_name])

        prompts = []
        responses = []

        for example in chat_threads_test_ds.select(range(1)):
            text = tokenizer.apply_chat_template(
                add_generation_prompt=True,
                documents=json.loads(example["documents"]),
                conversation=json.loads(example["messages"])[0:-1],
                tools=json.loads(example["tools"]),
                tokenize=False,
            )

            inputs = tokenizer(text, return_tensors="pt")

            streamer = TextStreamer(tokenizer, skip_prompt=True)

            token_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=512)

            response = tokenizer.decode(
                token_ids[0][len(inputs["input_ids"][0]) :],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
            )

            prompts.append(text)
            responses.append(response)

        prompt = prompts[0]
        response = responses[0]

        signature = mlflow.models.infer_signature(
            model_input=prompt,
            model_output=response,
        )

        model_info = mlflow.transformers.log_model(
            artifact_path="tuned_model",
            registered_model_name="TinyLlama-1.1B-Chat-v1.0-sft-chat_threads",
            signature=signature,
            task="text-generation",
            transformers_model={"model": trainer.model, "tokenizer": trainer.tokenizer},
        )

        return model_info.model_uri


def evaluate_model(
    chat_threads_train_ds: Dataset,
    chat_threads_validation_ds: Dataset,
    chat_threads_test_ds: Dataset,
    model_uri: str,
) -> pd.DataFrame:
    with mlflow.start_run(
        log_system_metrics=True,
        nested=True,
    ):

        components = mlflow.transformers.load_model(
            # device="cpu",
            model_uri=model_uri,
            return_type="components",
        )

        model, tokenizer = components["model"], components["tokenizer"]

        prompts = []
        responses = []

        train_ids = [5, 62]
        validation_ids = [25]
        test_ids = [0, 17]

        examples = concatenate_datasets(
            [
                chat_threads_train_ds.select(train_ids),
                chat_threads_validation_ds.select(validation_ids),
                chat_threads_test_ds.select(test_ids),
            ]
        )

        # for example in test_ds.select(range(1)):
        for example in examples:
            text = tokenizer.apply_chat_template(
                add_generation_prompt=True,
                documents=json.loads(example["documents"]),
                conversation=json.loads(example["messages"])[0:-1],
                tools=json.loads(example["tools"]),
                tokenize=False,
            )

            inputs = tokenizer(text, return_tensors="pt")

            streamer = TextStreamer(tokenizer, skip_prompt=True)

            token_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=512)

            response = tokenizer.decode(
                token_ids[0][len(inputs["input_ids"][0]) :],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
            )

            prompts.append(text)
            responses.append(response)

        df = pd.DataFrame(
            {
                "prompt": prompts,
                "response": responses,
            }
        )

        return df


def train_model_v2(
    chat_threads_train_ds: Dataset,
    chat_threads_validation_ds: Dataset,
    model_config: Dict,
    sft_config: Dict,
    sft_script_arguments: Dict,
) -> str:
    """Trains the model.

    Args:
        chat_threads_train_ds: Training data.
        chat_threads_validation_ds: Validation data.
        chat_threads_test_ds: Test data.
        model_config: Model configuration.
        sft_config: SFT configuration.
        sft_script_arguments: SFT script arguments.
        pretrained_model_uri: URI of the pretrained model.

    Returns:
        URI of the trained model.
    """
    model_config = ModelConfig(**model_config)
    sft_config = SFTConfig(**sft_config, hub_token=_get_hf_token())
    sft_script_arguments = SFTScriptArguments(**sft_script_arguments)

    print("model_config:")
    pprint(model_config)

    print("sft_config:")
    pprint(sft_config)

    print("sft_script_arguments:")
    pprint(sft_script_arguments)

    max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/tinyllama-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = "unsloth/tinyllama-bnb-4bit", # "unsloth/tinyllama" for 16bit loading
        # model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_name=fourbit_models[0],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )


def evaluate_model_v2(
    chat_threads_train_ds: Dataset,
    chat_threads_validation_ds: Dataset,
    chat_threads_test_ds: Dataset,
    model_uri: str,
) -> pd.DataFrame:
    with mlflow.start_run(
        log_system_metrics=True,
        nested=True,
    ):

        components = mlflow.transformers.load_model(
            # device="cpu",
            model_uri=model_uri,
            return_type="components",
        )

        model, tokenizer = components["model"], components["tokenizer"]

        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        prompts = []
        responses = []

        train_ids = [5, 62]
        validation_ids = [25]
        test_ids = [0, 17]

        examples = concatenate_datasets(
            [
                chat_threads_train_ds.select(train_ids),
                chat_threads_validation_ds.select(validation_ids),
                chat_threads_test_ds.select(test_ids),
            ]
        )

        # for example in test_ds.select(range(1)):
        for example in examples:
            # text = model["tokenizer"].apply_chat_template(
            # text = tokenizer.apply_chat_template(
            #     add_generation_prompt=True,
            #     documents=json.loads(example["documents"]),
            #     conversation=json.loads(example["messages"])[0:-1],
            #     tools=json.loads(example["tools"]),
            #     tokenize=False,
            # )

            # # inputs = model["tokenizer"](text, return_tensors="pt")
            # inputs = tokenizer(text, return_tensors="pt")

            # # streamer = TextStreamer(model["tokenizer"], skip_prompt=True)
            # streamer = TextStreamer(tokenizer, skip_prompt=True)

            # # token_ids = model["model"].generate(
            # token_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=512)

            # # response = model["tokenizer"].decode(
            # response = tokenizer.decode(
            #     # token_ids[0],
            #     token_ids[0][len(inputs["input_ids"][0]) :],
            #     clean_up_tokenization_spaces=True,
            #     skip_special_tokens=False,
            # )

            # prompts.append(text)
            # responses.append(response)

            inputs = tokenizer.apply_chat_template(
                add_generation_prompt=True,
                documents=json.loads(example["documents"]),
                conversation=json.loads(example["messages"])[0:-1],
                tools=json.loads(example["tools"]),
                return_tensors="pt",
                tokenize=True,
            ).to("cuda")

            outputs = model.generate(
                # input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.5, min_p=0.1
                input_ids=inputs,
                max_new_tokens=256,
                use_cache=True,
                temperature=0.0,
            )
            batch_decocded_outputs = tokenizer.batch_decode(outputs)

            text = batch_decocded_outputs[0][0 : len(tokenizer.decode(inputs[0]))]
            response = batch_decocded_outputs[0][len(tokenizer.decode(inputs[0])) :]

            prompts.append(text)
            responses.append(response)

        df = pd.DataFrame(
            {
                "prompt": prompts,
                "response": responses,
            }
        )

        return df
