import torch
from variables import shortcut_model_name2full_model_name
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, setup_chat_format
import os
import warnings
import argparse
import zipfile
import json

def finetune(subtask:str, shortcut_model_name:str, epochs:int, batch_size:int):
    
    assert shortcut_model_name in supported_shortcut_model_names
    assert subtask in supported_subtasks
    
    full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
    output_dir = f"finetuned_models/{subtask}/{shortcut_model_name}"

    ## prepare DATASET
    dataset = load_dataset(f"lavallone/{subtask}_semcor", split="train")
    def create_conversation(sample):
        return {
            "messages": [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]}
            ]
        }
    dataset = dataset.shuffle() #.select(range(100000))
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    data = dataset.train_test_split(test_size=0.1)

    ## prepare TOKENIZER
    if shortcut_model_name == "mistral": tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True, legacy=False)
    else: tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ## prepare MODEL
    # quantization step
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # different attn_implementation is required for each different model
    if shortcut_model_name == "phi_mini":
        model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2")
    elif shortcut_model_name == "gemma_2b" or shortcut_model_name == "gemma_9b":
        model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16, quantization_config=bnb_config, device_map="auto", attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16, quantization_config=bnb_config, device_map="auto") 
    model.config.use_cache = False

    ## setup lora configuration
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.1,
        r=256,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    ## TRAIN
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_total_limit=1,
        evaluation_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False,}
    )

    trainer.train()
    

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    supported_subtasks = ["selection", "generation"]
    supported_shortcut_model_names = ["llama_1b",
                                       "gemma_2b",
                                       "llama_3b",
                                       "phi_mini",
                                       "mistral",
                                       "phi_small",
                                       "llama_8b",
                                       "gemma_9b"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-st", type=str, help="Input the task")
    parser.add_argument("--shortcut_model_name", "-m", type=str, help="Input the model")
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", "-bs", type=int, help="Batch size")
    args = parser.parse_args()
    
    finetune(args.subtask, args.shortcut_model_name, args.epochs, args.batch_size)