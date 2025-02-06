from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from variables import shortcut_model_name2full_model_name, chat_template_prompts
from tqdm import tqdm
import warnings
import argparse
import json
import torch
import os
from openai import OpenAI
from utils import _get_gold_data, _create_folder, _print_log


def _generate_prompt(instance:dict, subtask:str, approach:str):
    """
    Generates a prompt based on the instance, subtask and approach.

    Args:
        instance (dict): The instance containing information about the task.
        subtask (str): The subtask of the instance, e.g., "selection", "generation", or "wic".
        approach (str): The approach used for generating the prompt.

    Returns:
        str or None: The generated prompt if applicable, otherwise None.
    """
        
    word = instance["word"]
    text = instance["text"].replace(" ,", ",").replace(" .", ".")
    candidate_definitions = "\n".join([f"{idx+1}) {x}" for idx, x in enumerate(instance["definitions"])])
    
    # we use chat_template
    chat_template_prompt_dict = chat_template_prompts[subtask][approach]
    if approach == "zero_shot": 
        prompt = [{"role": "user", "content": ""}]
        prompt[0]["content"] = chat_template_prompt_dict["prompt"].format(word=word,
                                                                            text=text,
                                                                            candidate_definitions=candidate_definitions)
    elif approach == "one_shot": 
        prompt = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}]
        prompt[0]["content"] = chat_template_prompt_dict["example"]
        prompt[1]["content"] = chat_template_prompt_dict["example_output"]
        prompt[2]["content"] = chat_template_prompt_dict["prompt"].format(word=word,
                                                                            text=text,
                                                                            candidate_definitions=candidate_definitions)
    elif approach == "few_shot": 
        prompt = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}]
        prompt[0]["content"] = chat_template_prompt_dict["example_1"]
        prompt[1]["content"] = chat_template_prompt_dict["example_1_output"]
        prompt[2]["content"] = chat_template_prompt_dict["example_2"]
        prompt[3]["content"] = chat_template_prompt_dict["example_2_output"]
        prompt[4]["content"] = chat_template_prompt_dict["example_3"]
        prompt[5]["content"] = chat_template_prompt_dict["example_3_output"]
        prompt[6]["content"] = chat_template_prompt_dict["prompt"].format(word=word,
                                                                            text=text,
                                                                            candidate_definitions=candidate_definitions)

    return prompt

def _prepare_finetuned_model(shortcut_model_name:str, checkpoint_path:str):
    # load the original model first
    full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
    if shortcut_model_name == "mistral": tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True, legacy=False)
    else: tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if shortcut_model_name == "phi_mini":
        base_model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
    else:
        base_model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda()
    
    # merge fine-tuned weights with the base model
    peft_model_id = checkpoint_path
    peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
    peft_model = peft_model.merge_and_unload()
    return tokenizer, peft_model

def process(subtask:str, approach:str, shortcut_model_name:str, is_finetuned:bool, checkpoint_path:str):
    """
    Processes the evaluation task for a specific subtask, approach, and model. Selection and generation subtasks only.

    Args:
        output_file_path (str): The path of the output folder.
        subtask (str): The subtask of the evaluation.
        approach (str): The approach used for evaluation.
        shortcut_model_name (str): The name of the model.
        is_finetuned (bool): If the model is finetuned or not.
        checkpoint_path (str): The path of the finetuned checkpoint.

    Returns:
        None
    """
    # we create folder structure
    output_file_path = _create_folder(subtask, approach, shortcut_model_name, is_finetuned)

    gold_data = _get_gold_data(subtask)
    n_instances_processed = 0
    json_data = []

    # if the model is finetuned, the checkpoint path is needed
    if is_finetuned:
        tokenizer, model = _prepare_finetuned_model(shortcut_model_name, checkpoint_path)
        pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)
    else:
        if shortcut_model_name == "gpt": global OPEN_AI_CLIENT; OPEN_AI_CLIENT = OpenAI()
        else:
            full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
            if shortcut_model_name == "mistral": tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True, legacy=False)
            else: tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            if shortcut_model_name == "phi_mini": model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
            else: model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda()
            pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)
    
    with open(f"{output_file_path}/output.txt", "a") as fa_txt, open(f"{output_file_path}/output.json", "w") as fw_json:
        for instance in tqdm(gold_data, total=len(gold_data)):

            n_instances_processed += 1
            instance_id = instance["id"]
            
            chat_prompt = _generate_prompt(instance, subtask, approach)
            if shortcut_model_name == "gpt":
                completion = OPEN_AI_CLIENT.chat.completions.create(
                                model=shortcut_model_name2full_model_name[shortcut_model_name],
                                messages=chat_prompt
                                )
                answer =  completion.choices[0].message.content
            else:
                prompt_template = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
                answer = pipe(prompt_template)[0]["generated_text"].replace(prompt_template, "").replace("\n", "").strip()
            
            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)

    # LOG
    last_prompt = chat_prompt if shortcut_model_name == "gpt" else prompt_template
    if args.log_config:
        if is_finetuned: shortcut_model_name = f"finetuned_{shortcut_model_name}"
        _print_log(subtask, approach, shortcut_model_name, last_prompt, n_instances_processed)

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-st", type=str, help="Input the task")
    parser.add_argument("--approach", "-a", type=str, help="Input the approach")
    parser.add_argument("--shortcut_model_name", "-m", type=str, help="Input the model")
    parser.add_argument("--is_finetuned", "-f", type=bool, default=False, help="If the model we want to test is finetuned or not")
    parser.add_argument("--checkpoint_path", "-cp", type=str, default=None, help="Input the checkpoint path")
    parser.add_argument("--log_config", "-l", type=bool, default=True, help="Log the results")
    args = parser.parse_args()

    supported_subtasks = ["selection", "generation"]
    supported_approaches = ["zero_shot", "one_shot", "few_shot"]
    supported_shortcut_model_names = ["llama_1b",
                                       "gemma_2b",
                                       "llama_3b",
                                       "phi_mini",
                                       "mistral",
                                       "phi_small",
                                       "llama_8b",
                                       "gemma_9b",
                                       "gpt"]
    assert args.shortcut_model_name in supported_shortcut_model_names
    assert args.subtask in supported_subtasks
    assert args.approach in supported_approaches
    # if we want to test a finetuned model we need to provide the checkpoint
    assert args.is_finetuned==False or args.checkpoint_path!=None

    process(args.subtask, args.approach, args.shortcut_model_name, args.is_finetuned, args.checkpoint_path)