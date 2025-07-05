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

import os
import re
import json
import random
import logging
from dataclasses import dataclass, field

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ['format', 'json_consistency', 'json_structure', 'f1_ents', 'f1_rels'],
        metadata={"help": "List of reward functions. Possible values: 'format', 'json_consistency', etc."},
    )
    entity_key: str = field(
        default="entities",
        metadata={"help": "JSON key for entities in output"},
    )
    relation_key: str = field(
        default="relations",
        metadata={"help": "JSON key for relations in output"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )
    dataset_name: str = field(
        default="dataset.json",
        metadata={"help": "Path to the dataset file"},
    )

def extract_json_from_text(text):
    """
    Extract valid JSON objects from text with debug logging.
    """
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
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON at position {idx}: {e}")
    return extracted_jsons

def validate_json_structure(data, entity_key, relation_key):
    """
    Validate JSON structure with configurable keys.
    """
    required_keys = {entity_key, relation_key}
    entity_required = {"id", "text", "type"}
    relation_required = {"head", "tail", "type"}

    if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
        logger.debug(f"JSON does not contain required keys: {required_keys}")
        return False

    if not isinstance(data[entity_key], list):
        logger.debug(f"Entities key '{entity_key}' is not a list")
        return False

    for entity in data[entity_key]:
        if not isinstance(entity, dict) or not entity_required.issubset(entity.keys()):
            logger.debug(f"Entity missing required keys: {entity_required}")
            return False
        if not all(isinstance(entity[k], t) for k, t in 
                   [("id", int), ("text", str), ("type", str)]):
            logger.debug(f"Entity has invalid types for keys: {entity}")
            return False

    if not isinstance(data[relation_key], list):
        logger.debug(f"Relations key '{relation_key}' is not a list")
        return False

    for relation in data[relation_key]:
        if not isinstance(relation, dict) or not relation_required.issubset(relation.keys()):
            logger.debug(f"Relation missing required keys: {relation_required}")
            return False
        if not all(isinstance(relation[k], str) for k in ["head", "tail", "type"]):
            logger.debug(f"Relation has invalid types for keys: {relation}")
            return False

    return True

def compute_f1(pred_list, true_list):
    """
    Compute F1 score between prediction and true lists.
    """
    tp = len(set(pred_list) & set(true_list))
    fp = len(set(pred_list) - set(true_list))
    fn = len(set(true_list) - set(pred_list))

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

def get_entities_f1_score(pred_entities, true_entities):
    """
    Compute F1 score for entities.
    """
    pred_list = {f"{entity['text']}_{entity['type']}" for entity in pred_entities}
    true_list = {f"{entity['text']}_{entity['type']}" for entity in true_entities}
    return compute_f1(pred_list, true_list)

def get_relations_f1_score(pred_relations, true_relations):
    """
    Compute F1 score for relations.
    """
    pred_list = {f"{rel['head']}_{rel['tail']}_{rel['type']}" for rel in pred_relations}
    true_list = {f"{rel['head']}_{rel['tail']}_{rel['type']}" for rel in true_relations}
    return compute_f1(pred_list, true_list)

def format_reward(completions, **kwargs):
    """
    Reward function that checks if the completion has a specific format.
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [("<think>" in content and "</think>" in content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def json_consistency_reward(completions, solution, **kwargs):
    """
    Reward function for JSON consistency.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [0.1 if len(extract_json_from_text(c)) == 1 else 0.0 for c in contents]

def json_structure_reward(completions, solution, **kwargs):
    """
    Reward function for JSON structure validation.
    """
    entity_key = kwargs.get('entity_key', 'entities')
    relation_key = kwargs.get('relation_key', 'relations')
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in contents:
        jsons = extract_json_from_text(content)
        if len(jsons) == 1:
            rewards.append(0.1 if validate_json_structure(jsons[0], entity_key, relation_key) else 0.0)
        else:
            rewards.append(0.0)
    return rewards

def f1_entities_reward(completions, solution, **kwargs):
    """
    Reward function for F1 score of entities.
    """
    entity_key = kwargs.get('entity_key', 'entities')
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        pred_jsons = extract_json_from_text(content)
        true_jsons = extract_json_from_text(sol)
        if len(pred_jsons) == 1 and len(true_jsons) == 1:
            try:
                pred_ents = pred_jsons[0].get(entity_key, [])
                true_ents = true_jsons[0].get(entity_key, [])
                rewards.append(get_entities_f1_score(pred_ents, true_ents))
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def f1_relations_reward(completions, solution, **kwargs):
    """
    Reward function for F1 score of relations.
    """
    relation_key = kwargs.get('relation_key', 'relations')
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        pred_jsons = extract_json_from_text(content)
        true_jsons = extract_json_from_text(sol)
        if len(pred_jsons) == 1 and len(true_jsons) == 1:
            try:
                pred_rels = pred_jsons[0].get(relation_key, [])
                true_rels = true_jsons[0].get(relation_key, [])
                rewards.append(get_relations_f1_score(pred_rels, true_rels))
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

reward_funcs_registry = {
    "format": format_reward,
    "json_consistency": json_consistency_reward,
    "json_structure": json_structure_reward,
    "f1_ents": f1_entities_reward,
    "f1_rels": f1_relations_reward,
}

def main(script_args, training_args, model_args):
    # Get reward functions with kwargs
    reward_funcs = []
    for func in script_args.reward_funcs:
        if func in reward_funcs_registry:
            reward_funcs.append(reward_funcs_registry[func])
        else:
            logger.warning(f"Unknown reward function '{func}' skipped")

    # Load the dataset
    dataset_path = script_args.dataset_name
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' does not exist.")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    for idx, example in enumerate(dataset):
        if 'prompt' not in example or 'solution' not in example:
            raise ValueError(f"Dataset entry {idx} missing required keys")
    random.seed(script_args.seed)
    random.shuffle(dataset)
    split = int(len(dataset) * 0.8)
    train_data = dataset[:split]
    test_data = dataset[split:]

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        reward_kwargs={
            'entity_key': script_args.entity_key,
            'relation_key': script_args.relation_key,
        }
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
