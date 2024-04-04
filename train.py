#    Modification Copyright 2024 Zhenyu He
#    Modification Copyright 2023 Dawei Zhu
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import random
import os
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
import deepspeed
from config_llama import MyLlamaConfig
from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig, default_data_collator, AutoTokenizer
from datasets import load_dataset, load_from_disk

from my_modeling_llama_bipe_rope import MyLlamaForCausalLM as MyLlamaForCausalLM_bipe_rope
from my_modeling_llama_bipe_alibi import MyLlamaForCausalLM as MyLlamaForCausalLM_bipe_alibi

transformers.logging.set_verbosity_info()

@dataclass
class ModelArguments:
    config_name: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_cache_dir: str = field(default=None, metadata={"help": "Path to the data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_position_embeddings: int = field(
        default=1024,
        metadata={"help": "Maximum position embeddings."},
    )
    rope_scaling_type: Optional[str] = field(default=None)
    rope_scaling_factor: float = field(default=1.0)
    resume_from_checkpoint: Optional[bool] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

              
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.config_name:
        config = MyLlamaConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = MyLlamaConfig.from_pretrained(model_args.model_name_or_path)
    else:
        raise NotImplementedError

    scaled_max_position_embeddings=int(training_args.model_max_position_embeddings * training_args.rope_scaling_factor)
    config.max_position_embeddings=scaled_max_position_embeddings

    if training_args.rope_scaling_type is not None:
        config.rope_scaling={"type": training_args.rope_scaling_type, "factor": training_args.rope_scaling_factor}
        if training_args.rope_scaling_type == "yarn":
            config.rope_scaling["original_max_position_embeddings"] = training_args.model_max_position_embeddings
        
    if config.rpe_type == "bipe_rope" or config.rpe_type == "rope":
        LlamaForCausalLM = MyLlamaForCausalLM_bipe_rope
    elif config.rpe_type == "bipe_alibi" or config.rpe_type == "alibi":
        LlamaForCausalLM = MyLlamaForCausalLM_bipe_alibi
    else:
        raise NotImplementedError

    if model_args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
    else:
        model = LlamaForCausalLM(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        if training_args.local_rank == 0:
            print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    tokenizer = AutoTokenizer.from_pretrained(
        "llama_tokenizer",
        use_fast=True,
    )

    raw_datasets = load_from_disk(data_args.dataset_cache_dir)
    # raw_datasets = load_dataset('monology/pile-uncopyrighted')

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
     

    if training_args.local_rank > 0: 
        torch.distributed.barrier()

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=64,
        remove_columns=column_names,
        load_from_cache_file=True,
        cache_file_names={"train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train.arrow",\
            "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation.arrow", \
            "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test.arrow"},
        desc="Running tokenizer on dataset",
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // config.train_scale) * config.train_scale
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + config.train_scale] for i in range(0, total_length, config.train_scale)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(f"{data_args.dataset_cache_dir}/{config.train_scale}", exist_ok=True)
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=64,
        load_from_cache_file=True,
        cache_file_names={"train": f"{data_args.dataset_cache_dir}/{config.train_scale}/lm_datasets_train.arrow",\
            "validation": f"{data_args.dataset_cache_dir}/{config.train_scale}/lm_datasets_validation.arrow", \
            "test": f"{data_args.dataset_cache_dir}/{config.train_scale}/lm_datasets_test.arrow"},
        desc=f"Grouping texts in chunks of {config.train_scale}",
    )


    # if training_args.local_rank == 0:
    print(f"rank{training_args.local_rank} loading datasets")

    # if training_args.local_rank == 0:
    print(f"rank{training_args.local_rank} datasets loaded")

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]


    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print("len(train_dataset):", len(train_dataset))
        # for index in random.sample(range(len(train_dataset)), 3):
        #     print(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    data_collator = default_data_collator # DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_module = dict(train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator)

    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.do_eval:
        logging.info("*** Evaluate on valid set***")
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
    if training_args.do_predict:
        logging.info("*** Evaluate on test set***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)




if __name__ == "__main__":
    train()
