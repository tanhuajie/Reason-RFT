# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Type
from functools import partial

from datasets import load_dataset

from stage_rl.configs import GRPOConfig
from stage_rl.trainer import MultiModalGRPOTrainer
from stage_rl.prompt import *
from stage_rl.reward import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config


logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reason', 'length'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'reason', 'length'"},
    )
    trainer_cls: Type = field(
        default=MultiModalGRPOTrainer, 
        metadata={"help": "Trainer class"}
    )
    use_vllm_for_gen: str = field(
        default="true", 
        metadata={"help": "Whether to use vllm for fast generation"}
    )
    use_system_prompt: str = field(
        default="false", 
        metadata={"help": "Whether to use system_prompt (True) or use question_template instead (False)"}
    )
    task_name: str = field(
        default="trance", 
        metadata={"help": "Task name: ['trance', trance-only-full', 'trance-full-caption', 'clevr-math', 'geoqa']"}
    )
    image_path: Optional[str] = field(
        default="/home/vlm/train_images/", 
        metadata={"help": "Path to images"}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def main(script_args, training_args, model_args):

    use_system_prompt = False if script_args.use_system_prompt == "false" else True
    use_vllm_for_gen = False if script_args.use_vllm_for_gen == "false" else True

    assert script_args.task_name in ['trance', 'trance-only-full', 'trance-full-caption', 'trance-penalty', 'clevr-math', 'geoqa'], f"Task ({script_args.task_name}) is not supported."

    if script_args.task_name == "trance":
        SYSTEM_PROMPT = TRANCE_SYSTEM_PROMPT
        QUESTION_PROMPT = TRANCE_QUESTION_PROMPT
        assert "length" not in script_args.reward_funcs, f"Length reward is not supported in trance Task."
        reward_funcs_registry = {
            "accuracy": func_accuracy_reward,
            "format": format_reward,
            "reason": reasoning_steps_reward,
        }
    elif script_args.task_name == "trance-only-full":
        SYSTEM_PROMPT = TRANCE_SYSTEM_PROMPT
        QUESTION_PROMPT = TRANCE_QUESTION_PROMPT
        assert "length" not in script_args.reward_funcs, f"Length reward is not supported in trance Task."
        reward_funcs_registry = {
            "accuracy": only_full_func_accuracy_reward,
            "format": format_reward,
            "reason": reasoning_steps_reward,
        }
    elif script_args.task_name == "trance-full-caption":
        SYSTEM_PROMPT = TRANCE_SYSTEM_CAPTION_PROMPT
        QUESTION_PROMPT = TRANCE_QUESTION_CAPTION_PROMPT
        assert "length" not in script_args.reward_funcs, f"Length reward is not supported in trance Task."
        reward_funcs_registry = {
            "accuracy": only_full_func_accuracy_reward,
            "format": caption_format_reward,
            "reason": reasoning_steps_reward,
        }
    elif script_args.task_name == "trance-penalty":
        SYSTEM_PROMPT = TRANCE_SYSTEM_PROMPT
        QUESTION_PROMPT = TRANCE_QUESTION_PROMPT
        assert "length" not in script_args.reward_funcs, f"Length reward is not supported in trance Task."
        reward_funcs_registry = {
            "accuracy": penalty_func_accuracy_reward,
            "format": format_reward,
            "reason": reasoning_steps_reward,
        }
    elif script_args.task_name == "clevr-math":
        SYSTEM_PROMPT = CLEVR_MATH_SYSTEM_PROMPT
        QUESTION_PROMPT = CLEVR_MATH_QUESTION_PROMPT
        reward_funcs_registry = {
            "accuracy": accuracy_reward,
            "format": format_reward,
            "reason": reasoning_steps_reward,
            "length": len_reward,
        }
    elif script_args.task_name == "geoqa":
        SYSTEM_PROMPT = GEOQA_SYSTEM_PROMPT
        QUESTION_PROMPT = GEOQA_QUESTION_PROMPT
        reward_funcs_registry = {
            "accuracy": accuracy_reward, # math_accuracy_reward,
            "format": format_reward,
            "reason": reasoning_steps_reward,
            "length": len_reward,
        }
    else:
        SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT
        QUESTION_PROMPT = GENERAL_QUESTION_PROMPT
        reward_funcs_registry = {
            "accuracy": accuracy_reward,
            "format": format_reward,
            "reason": reasoning_steps_reward,
            "length": len_reward,
        }

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    if script_args.dataset_name.endswith('.json'):
        dataset = load_dataset('json', data_files=script_args.dataset_name)
        # Format into conversation (multi-image / single-image / text-only)
        def make_conversation(example, image_path=None, use_system_prompt=False):
            # multimodal sample
            if "image" in example and example["image"]:
                if isinstance(example["image"], list):
                    images = []
                    for item in example["image"]:
                        if isinstance(item, str):
                            images.append(os.path.join(image_path, item))
                        elif isinstance(item, dict):
                            images.append(os.path.join(image_path, item["path"]))
                        else:
                            raise TypeError("Unsupported Format.")
                elif isinstance(example["image"], str):
                    images = [os.path.join(image_path, example["image"])]
                elif isinstance(example["image"], dict):
                    images.append(os.path.join(image_path, example["image"]["path"]))
                else:
                    raise TypeError("Unsupported Format.")
                
                if use_system_prompt:
                    return {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    *[{"type": "image"} for _ in images],
                                    {"type": "text", "text": example["problem"]},
                                ],
                            },
                        ],
                        "image": images
                    }
                else:
                    return {
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    *[{"type": "image"} for _ in images],
                                    {"type": "text", "text": QUESTION_PROMPT.format(Question=example["problem"])},
                                ],
                            },
                        ],
                        "image": images
                    }
                
            # text-only sample
            else:
                if use_system_prompt:
                    return {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": example["problem"]},
                        ],
                    }
                else:
                    return {
                        "prompt": [
                            {"role": "user", "content": QUESTION_PROMPT.format(Question=example["problem"])},
                        ],
                    }
        
        dataset = dataset.map(partial(make_conversation, image_path=script_args.image_path, use_system_prompt=use_system_prompt))
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        # Format into conversation (single-image / text-only)
        def make_conversation_hf(example):
            # multimodal sample
            if "image" in example: # BUG Note: Not yet support for mix multimodal and text-only AND multi-images
                if use_system_prompt:
                    return {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": example["problem"]},
                                ],
                            },
                        ],
                    }
                else:
                    return {
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": QUESTION_PROMPT.format(Question=example["problem"])},
                                ],
                            },
                        ],
                    }
            # text-only sample
            else:
                if use_system_prompt:
                    return {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": example["problem"]},
                        ],
                    }
                else:
                    return {
                        "prompt": [
                            {"role": "user", "content": QUESTION_PROMPT.format(Question=example["problem"])},
                        ],
                    }
        dataset = dataset.map(make_conversation_hf)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # Initialize the GRPO trainer
    trainer = script_args.trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        use_vllm_for_gen=use_vllm_for_gen
    )

    # Train model
    trainer.train()
    trainer.save_model(training_args.output_dir)

    # Save and push to hub
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
