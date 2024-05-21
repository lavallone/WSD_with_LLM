import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
import sys
import os

import warnings
warnings.filterwarnings("ignore")

#selection vs generation
task = "selection"

dataset_name = f"../data/training/{task}/training.json"
data = load_dataset("json", data_files=dataset_name)
data = data["train"].train_test_split(test_size=0.1)

model_name = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_model = model

lora_alpha = 32
lora_dropout = 0.05
lora_rank = 32

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    #target_modules=["q_proj", "v_proj"],
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
)

peft_model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

output_dir = "output_dir"

per_device_train_batch_size = 16 #4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
#save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 180 #100 #500
warmup_ratio = 0.03
lr_scheduler_type = "cosine" #"constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    #save_steps=save_steps,
    save_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    #max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    push_to_hub=False,
    num_train_epochs=10,
    save_total_limit=1,
    evaluation_strategy="epoch",
    load_best_model_at_end=True
)

max_seq_length = 2048

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=data['train'],
    eval_dataset=data['test'],
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

peft_model.config.use_cache = False

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.bfloat16)

trainer.train()
