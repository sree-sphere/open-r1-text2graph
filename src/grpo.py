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

import re
import json
import random
from dataclasses import dataclass, field

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ['format', 'json_consistency', 'json_structure', 'f1_ents', 'f1_rels'],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )

def extract_json_from_text(text):
    json_start = 0
    json_end = 0
    close_brace_count = 0
    extracted_jsons = []
    for idx, char in enumerate(text):
        if char == '{':
            if close_brace_count == 0:
                json_start = idx
            close_brace_count += 1
        elif char == '}':
            close_brace_count -= 1
            if close_brace_count == 0:
                json_end = idx + 1
                extracted_json = text[json_start:json_end]
                try:
                    extracted_jsons.append(json.loads(extracted_json))
                except json.JSONDecodeError:
                    pass
    return extracted_jsons

def validate_json_structure(data):
    required_keys = {"entities", "relations"}
    entity_required_keys = {"id", "text", "type"}
    relation_required_keys = {"head", "tail", "type"}

    if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
        return False

    if not isinstance(data["entities"], list):
        return False

    for entity in data["entities"]:
        if not isinstance(entity, dict) or not entity_required_keys.issubset(entity.keys()):
            return False
        if not isinstance(entity["id"], int) or not isinstance(entity["text"], str) or not isinstance(entity["type"], str):
            return False

    if not isinstance(data["relations"], list):
        return False

    for relation in data["relations"]:
        if not isinstance(relation, dict) or not relation_required_keys.issubset(relation.keys()):
            return False
        if not isinstance(relation["head"], str) or not isinstance(relation["tail"], str) or not isinstance(relation["type"], str):
            return False

    return True

def compute_f1(pred_list, true_list):
    tp = len(set(pred_list) & set(true_list))
    fp = len(set(pred_list) - set(true_list))
    fn = len(set(true_list) - set(pred_list))

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

def get_entities_f1_score(pred_entities, true_entities):
    pred_list = {f"{entity['text']}_{entity['type']}" for entity in pred_entities}
    true_list = {f"{entity['text']}_{entity['type']}" for entity in true_entities}
    f1 = compute_f1(pred_list, true_list)
    return f1

def get_relations_f1_score(pred_relations, true_relations):
    pred_list = {f"{rel['head']}_{rel['tail']}_{rel['type']}" for rel in pred_relations}
    true_list = {f"{rel['head']}_{rel['tail']}_{rel['type']}" for rel in true_relations}
    f1 = compute_f1(pred_list, true_list)
    return f1

def json_consistency_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        extracted_jsons = extract_json_from_text(content)
        if len(extracted_jsons)==1:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards

def json_structure_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        extracted_jsons = extract_json_from_text(content)
        if len(extracted_jsons)==1:
            extracted_json = extracted_jsons[0]
            val = validate_json_structure(extracted_json)
            if val:
                rewards.append(0.1)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def f1_entities_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        extracted_jsons_pred = extract_json_from_text(content)
        extracted_jsons_true = extract_json_from_text(sol)

        if len(extracted_jsons_pred)==1 and len(extracted_jsons_true)==1:
            json_pred = extracted_jsons_pred[0]
            json_true = extracted_jsons_true[0]

            f1_reward = 0
            try:
                f1_reward += get_entities_f1_score(json_pred['entities'], json_true['entities'])
            except:
                pass
    
            rewards.append(f1_reward)
        else:
            rewards.append(0)
    return rewards

def f1_relations_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        extracted_jsons_pred = extract_json_from_text(content)
        extracted_jsons_true = extract_json_from_text(sol)

        if len(extracted_jsons_pred)==1 and len(extracted_jsons_true)==1:
            json_pred = extracted_jsons_pred[0]
            json_true = extracted_jsons_true[0]

            f1_reward = 0
            try:
                f1_reward += get_relations_f1_score(json_pred['relations'], json_true['relations'])
            except:
                pass
    
            rewards.append(f1_reward)
        else:
            rewards.append(0)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [("<think>" in content and "</think>" in content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "format": format_reward,
    "json_consistency": json_consistency_reward,
    "json_structure": json_structure_reward,
    "f1_ents": f1_entities_reward,
    "f1_rels": f1_relations_reward,

}

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    with open(script_args.dataset_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    random.shuffle(dataset)

    train_dataset = dataset[:int(len(dataset)*0.8)]
    test_dataset = dataset[int(len(dataset)*0.8):]

    print(train_dataset[0])

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
