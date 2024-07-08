import torch
from variables import shortcut_model_name2full_model_name
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
import os
import warnings
import argparse
import zipfile

def finetune(subtask:str, shortcut_model_name:str):
    
    assert shortcut_model_name in supported_shortcut_model_names
    assert subtask in supported_subtasks
    
    full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
    output_dir = f"finetuned_models/{subtask}/{shortcut_model_name}"

    ## prepare DATASET
    dataset_path = f"../data/training/{subtask}/training.json"
    if not os.path.exists(dataset_path):
        dir_path = f"../data/training/{subtask}/"
        file_to_unzip_name = "selection.semcor.zip" if subtask== "selection" else "generation.semcor.zip"
        file_to_unzip_path = os.path.join(dir_path, file_to_unzip_name)
        with zipfile.ZipFile(file_to_unzip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(dir_path))
        extracted_file_name = "data.semcor.json" if subtask== "selection" else "generation.semcor.json"
        extracted_file_path = os.path.join(dir_path, extracted_file_name)
        os.rename(extracted_file_path, os.path.join(dir_path, "training.json"))
    data = load_dataset("json", data_files=dataset_path)
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
        trust_remote_code=True
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
    

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    supported_subtasks = ["selection", "generation", "wic"]
    supported_shortcut_model_names = ["llama_2", "llama_3", "mistral", "falcon", "vicuna", 
                                      "tiny_llama", "stability_ai", "h2o_ai",
                                      "phi_3_small", "phi_3_mini", "gemma_2b", "gemma_9b"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-st", type=str, help="Input the task")
    parser.add_argument("--shortcut_model_name", "-m", type=str, help="Input the model")
    args = parser.parse_args()
    
    finetune(args.subtask, args.shortcut_model_name)