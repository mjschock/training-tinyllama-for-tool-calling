import json
import os
from pprint import pprint
from typing import Dict, List

import evaluate
import mlflow
import numpy as np
import torch
from datasets import load_dataset
from mlflow.models import set_model
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    ChatChoice,
    ChatMessage,
    ChatParams,
    ChatResponse,
    FunctionToolCallArguments,
    ToolCall,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import EvalPrediction
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

project_root = os.getcwd()

while not os.path.exists(os.path.join(project_root, "register_prefect_flow.py")):
    project_root = os.path.dirname(project_root)

print(f"Project root: {project_root}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = None
load_in_4bit = True
max_seq_length = 4096
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "data/06_models/model/lora"
# model_name = "data/06_models/mjschock/TinyLlama-1.1B-Chat-v1.0_lora_sft"
# model_name = f"{project_root}/data/06_models/mjschock/TinyLlama-1.1B-Chat-v1.0-tool-calling-sft/lora"
# model_name = f"{project_root}/data/06_models/mjschock/TinyLlama-1.1B-Chat-v1.0-tool-calling-sft/lora"

class ModelClient:
    def __init__(self):
        user_id = "mjschock" # TODO: get this dynamically
        pretrained_model_name = "TinyLlama-1.1B-Chat-v1.0"
        model_name = f"{project_root}/data/06_models/{user_id}/{pretrained_model_name}-tool-calling-sft/lora"
        # model_name = f"{project_root}/data/06_models/{user_id}/{pretrained_model_name}-tool-calling-sft/unsloth_lora" # TODO:Maybe this would work better for using the model rather than code path?

        model, tokenizer = FastLanguageModel.from_pretrained(
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            model_name=model_name,
        )

        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        self.model = model
        self.tokenizer = tokenizer

    def chat_completion_request(
        self,
        documents: list,
        messages: list,
        tools: list,
    ):
        # FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        inputs = self.tokenizer.apply_chat_template(
            add_generation_prompt=True,
            conversation=messages,
            documents=documents,
            return_tensors="pt",
            tokenize=True,
            tools=tools,
        ).to(device)

        outputs = self.model.generate(
            do_sample=False,
            input_ids=inputs,
            max_new_tokens=256,
            use_cache=True,
            # temperature=0.0,
        )

        batch_decoded_outputs = self.tokenizer.batch_decode(outputs)

        choices: List[ChatChoice] = []

        for i in range(len(batch_decoded_outputs)):
            response = batch_decoded_outputs[i][
                len(self.tokenizer.decode(inputs[i])) :
            ].replace(
                self.tokenizer.eos_token, ""
            )  # TODO: skip special tokens when decoding instead?

            try:
                response = json.loads(response)

                finish_reason: str = response.get("finish_reason")
                tool_calls_json = response.get("tool_calls")
                tool_calls: List[ToolCall] = []

                for tool_call_json in tool_calls_json:
                    tool_call = ToolCall(
                        function=FunctionToolCallArguments(
                            arguments=tool_call_json.get("arguments"),
                            name=tool_call_json.get("name"),
                        ),
                        id=tool_call_json.get("id"),
                        type="function",
                    )

                    tool_calls.append(tool_call)

                message: ChatMessage = ChatMessage(
                    role="assistant",
                    tool_calls=tool_calls,
                )

            except json.JSONDecodeError:
                finish_reason: str = "stop"
                message: ChatMessage = ChatMessage(
                    role="assistant",
                    content=response,
                )

            choices.append(
                ChatChoice(
                    index=i,
                    finish_reason=finish_reason,
                    logprobs=None,
                    message=message,
                )
            )

        return ChatResponse(
            choices=choices,
        )


class Agent(ChatModel):
    def __init__(self):
        # self.model_name = "llama3.2:1b"
        self.client = None

    def load_context(self, context):
        # self.model_name = "llama3.2:1b"
        # self.client = ollama.Client()
        print('=== load_context ===')
        print('context:', context)

        self.client = ModelClient()

    # the core method that needs to be implemented. this function
    # will be called every time a user sends messages to our model
    # @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, context, messages: list[ChatMessage], params: ChatParams):
        # instantiate the OpenAI client
        # client = OpenAI()

        # convert the messages to a format that the OpenAI API expects
        messages = [m.to_dict() for m in messages]

        print("params:")
        pprint(params.to_dict())

        tools = params.tools or []

        print("tools:")
        pprint(tools)

        tools = [t.to_dict() for t in tools]

        print("tools:")
        pprint(tools)

        # call the OpenAI API
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=messages,
        #     # pass the tools in the request
        #     tools=self.tools,
        # )

        response = self.client.chat_completion_request(
            documents=[],  # we don't need documents for this example
            messages=messages,
            # tools=self.tools,
            # tools=[],
            tools=tools,
        )

        # return the result as a ChatResponse, as this
        # is the expected output of the predict method
        return ChatResponse.from_dict(response.to_dict())


set_model(Agent())
