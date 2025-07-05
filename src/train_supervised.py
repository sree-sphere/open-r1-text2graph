import os
import json
import copy
import logging
import argparse
import random
from tqdm import tqdm

import bitsandbytes as bnb
import torch
from torch.utils.data import Dataset

from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define the dataset class
class Text2JSONDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        user_chat = item['prompt']
        output = item['solution']

        # Build full conversation
        full_chat = copy.deepcopy(user_chat)
        full_chat.append({"role": "assistant", "content": str(output)})

        # Tokenize user part and full conversation
        user_input = self.tokenizer.apply_chat_template(user_chat, tokenize=False)
        full_input = self.tokenizer.apply_chat_template(full_chat, tokenize=False)

        user_inputs = self.tokenizer(user_input, return_tensors="pt", truncation=True, padding=False)
        full_inputs = self.tokenizer(full_input, return_tensors="pt", max_length=self.max_length, 
                                   truncation=True, padding='max_length')

        user_input_ids = user_inputs["input_ids"].squeeze(0)
        full_input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)

        # Mask user tokens in labels
        labels = full_input_ids.clone()
        labels[:user_input_ids.shape[0]] = -100  # Mask user part
        labels[attention_mask.squeeze() == 0] = -100  # Mask padding

        return {
            'input_ids': full_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Helper function to find linear module names
def find_all_linear_names(model, quantize=False):
    cls = bnb.nn.Linear4bit if quantize else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Main function
def main(args):
    logger.info("Starting script with configuration: %s", args)

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path {args.data_path} does not exist.")
    if args.fp16 and args.bf16:  # needed for 16 bit
        raise ValueError("Cannot enable both FP16 and BF16. Choose one.")

    QUANTIZE = args.quantize
    USE_LORA = args.use_lora
    model_path = args.model_path

    # Device setup
    device_map = "auto" if torch.cuda.is_available() else None

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ) if QUANTIZE else None

    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else None,
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=args.hf_token,
        attn_implementation="flash_attention_2" if args.use_flash_attention else None
    )

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA setup
    if USE_LORA:
        modules = find_all_linear_names(model, quantize=QUANTIZE)
        logger.info(f"LoRA target modules: {modules}")
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )
    else:
        peft_config = None

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=args.hf_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.chat_template = CHAT_TEMPLATE

    # Load and shuffle data
    with open(args.data_path, encoding='utf-8') as f:
        data = json.load(f)
    random.seed(args.seed)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]
    
    train_dataset = Text2JSONDataset(train_data, tokenizer, max_length=args.max_length)
    test_dataset = Text2JSONDataset(test_data, tokenizer, max_length=args.max_length)

    logger.info("Dataset lengths - Train: %d, Test: %d", len(train_dataset), len(test_dataset))

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_32bit",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        save_total_limit=3,
        report_to="tensorboard" if args.log_dir else "none",
        logging_dir=args.log_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        max_seq_length=args.max_length,
        packing=False,
    )

    trainer.train()

    # Save merged model if using LoRA
    if peft_config is not None:
        logger.info("Merging LoRA weights into the base model.")
        trainer.model = trainer.model.merge_and_unload()
        merged_path = os.path.join(args.output_dir, 'merged')
        trainer.model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text2JSON Dataset Training Script")

    parser.add_argument('--model_path', type=str, required=True, help="Path to the model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the training dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save trained models.")
    parser.add_argument('--hf_token', type=str, required=False, default=None, help="Hugging Face authentication token.")
    parser.add_argument('--max_length', type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument('--num_train_epochs', type=int, default=2, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2, help="Training batch size.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument('--learning_rate', type=float, default=5e-6, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay.")
    parser.add_argument('--fp16', action='store_true', help="Enable FP16 training.")
    parser.add_argument('--bf16', action='store_true', help="Enable BF16 training.")
    parser.add_argument('--max_grad_norm', type=float, default=0.9, help="Maximum gradient norm.")
    parser.add_argument('--max_steps', type=int, default=-1, help="Maximum training steps.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument('--logging_steps', type=int, default=1, help="Logging steps.")
    parser.add_argument('--eval_steps', type=int, default=10000, help="Evaluation steps.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save steps.")
    parser.add_argument('--quantize', action='store_true', help="Enable quantization.")
    parser.add_argument('--use_lora', action='store_true', help="Enable LoRA training.")
    parser.add_argument('--use_flash_attention', action='store_true', help="Enable Flash Attention")
    parser.add_argument('--log_dir', type=str, default=None, help="TensorBoard logging directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for data splitting")

    args = parser.parse_args()

    main(args)
