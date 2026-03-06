# coding=utf-8
# Copyright 2023 The BigCode team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script for StarCoder chat model."""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

from config import DataArguments, ModelArguments
from dialogues import get_dialogue_template
from utils import create_datasets


@dataclass
class ScriptArguments:
    """Arguments for training script."""

    model_name: Optional[str] = field(default="bigcode/starcoder", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="HuggingFaceH4/oasst1_en", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(output_dir="./results")
    )

    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    )


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get dialogue template
    dialogue_template = get_dialogue_template(data_args.dialogue_template)

    # Set EOS token for dialogue template
    if dialogue_template.end_token is not None:
        tokenizer.eos_token = dialogue_template.end_token
        # Update pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    raw_datasets = create_datasets(tokenizer, data_args, training_args, dialogue_template)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    # Load model
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        load_in_8bit=model_args.load_in_8bit,
        trust_remote_code=True,
    )

    # Resize embeddings if needed
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_peft:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["c_proj", "c_attn", "q_attn"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(training_args.output_dir, "final_checkpoint/"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "final_checkpoint/"))


if __name__ == "__main__":
    train()
