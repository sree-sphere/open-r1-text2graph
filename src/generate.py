import re
import os
import json
import torch
import copy
import random
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ZERO_SHOT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an assistant trained to process any text and extract named entities and relations from it. "
            "Your task is to analyze user-provided text, identify all unique and contextually relevant entities, and infer meaningful relationships between them"
            "Additionaly you will get ready extracted results and your task is to generate thinking process as you would do not knowing the final JSON output."
            "You need to formulate your reasoning process and encapsulate it in <think> </think> tag, this is the only thing you return."
        ),
    },

]

ZERO_SHOT_PROMPT = """Analyze this text and JSON output and produce your thinking"""


def create_chat(prompt, text):
    return prompt.format(text)

def generate_response(llm, chats, sampling_params):
    responses = llm.generate(chats, sampling_params, use_tqdm=False)
    return responses

def process_input(example):
    messages =  copy.deepcopy(ZERO_SHOT_MESSAGES)
    text = messages[-1]['content']
    text+="Here is the JSON output:"
    solution = example['solution']
    text+=solution
    messages.append({
        "role": 'user',
        "content": text
    })
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def generate_dataset(examples, llm, sampling_params, batch_size=8, max_lines=None):
    batch_chats = []
    batch_examples = []
    text = True
    final_results = []
    for i in tqdm(range(0, max_lines)):
        example = examples[i]
        
        batch_examples.append(example)
        chat = process_input(example)
        batch_chats.append(chat)

        if len(batch_chats)==batch_size:
            try:
                responses = generate_response(llm, batch_chats, sampling_params)
            except Exception as err:
                print(err)
                continue

            batch_results = [ {'prompt': batch_examples[j]['prompt'], 
                                'solution': response.outputs[0].text +'\n'+batch_examples[j]['solution']}
                                            for j, response in enumerate(responses)]

            final_results.extend(batch_results)

            batch_chats = []
            batch_examples = []

    return final_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= "data/text2graph.json")
    parser.add_argument('--save_path', type=str, default= "data/text2graph_with_thinking.json")
    parser.add_argument('--model', type=str, default= "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument('--quantization', type=str, default= "fp8")
    parser.add_argument('--max_examples', type=int, default= 500)
    parser.add_argument('--batch_size', type=int, default= 8)
    parser.add_argument('--temperature', type=float, default= 0.75)
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        texts = json.load(f)
        random.shuffle(texts)
        print('Texts count: ', len(texts))

    llm = LLM(model=args.model,
                    max_model_len = 8129, 
                    tensor_parallel_size=1, dtype="half",
                        gpu_memory_utilization = 0.9, quantization = args.quantization)

    sampling_params = SamplingParams(temperature = args.temperature, repetition_penalty = 1.1, top_k=100, max_tokens=4096, top_p=0.8, stop="<end>")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = generate_dataset(texts, llm, sampling_params,
                                batch_size=args.batch_size, max_lines=args.max_examples)
    
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=1)