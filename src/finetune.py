import torch
from variables import shortcut_model_name2full_model_name
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
import sys
import os
import warnings
import argparse

def finetune(subtask:str, shortcut_model_name:str):
    
    assert shortcut_model_name in supported_shortcut_model_names
    assert subtask in supported_subtasks
    
    full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
    output_dir = f"finetuned_models/{subtask}/{shortcut_model_name}"

    ## prepare DATASET
    dataset_name = f"../data/training/{subtask}/training.json"
    data = load_dataset("json", data_files=dataset_name)
    data = data["train"].train_test_split(test_size=0.1)

    ## prepare TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    ## prepare MODEL
    # quantization step
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # setup lora configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    )
    peft_model = get_peft_model(model, peft_config)

    ## TRAIN
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        num_train_epochs=10,
        save_total_limit=1,
        evaluation_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        peft_config=peft_config,
        dataset_text_field="prompt",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    peft_model.config.use_cache = False

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.bfloat16)

    trainer.train()
    #trainer.push_to_hub()
    

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    supported_subtasks = ["selection", "generation"]
    supported_shortcut_model_names = ["llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2", "falcon-7b-instruct", "vicuna-7b-v1.5", 
                                      "microsoft-phi-1_5", "TinyLlama-TinyLlama-1.1B-Chat-v1.0", "stabilityai-stablelm-2-1_6b-chat", "h2oai-h2o-danube2-1.8b-chat",
                                      "microsoft-phi-2", "microsoft-phi-3-mini-128k-instruct", "meta-llama-Meta-Llama-3-8B",
                                      "openlm-research-open_llama_3b_v2", "openlm-research-open_llama_7b_v2"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-st", type=str, help="Input the task")
    parser.add_argument("--shortcut_model_name", "-m", type=str, help="Input the model")
    args = parser.parse_args()
    
    finetune(args.subtask, args.shortcut_model_name)