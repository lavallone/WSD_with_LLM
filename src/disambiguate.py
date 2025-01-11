from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from variables import shortcut_model_name2full_model_name, prompts, chat_template_prompts
from tqdm import tqdm
import warnings
import argparse
import time
import json
import torch
import os

def countdown(t):
    """
    Activates and displays a countdown timer in the terminal.

    Args:
        t (int): The duration of the countdown timer in seconds.

    Returns:
        None
    """
    while t > 0:
        mins, secs = divmod(t, 60)
        timer = '{:02d}'.format(secs)
        print(f"\033[1mWarning\033[0m: Found output files in the target directory! I will delete them in {timer}", end='\r')
        time.sleep(1)
        t -= 1

def _generate_prompt(instance:dict, subtask:str, approach:str, chat_template = False):
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
    
    # if we use chat_template, we need to deal with each approach differently
    if chat_template:
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

def _get_gold_data(subtask:str):
    """
    Loads gold data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    data = []
    if subtask in ["selection", "generation"]:
        data_path = "../data/evaluation/ALL_preprocessed.json"
        with open(data_path, "r") as json_file:
            data_ = json.load(json_file)
        data.append(data_)
    elif subtask == "wic":
        data_path = "../data/evaluation/test.en-en.data"
        gold_path = "../data/evaluation/test.en-en.gold"
        with open(data_path, "r") as fr:
            data_ = json.load(fr)
        with open(gold_path, "r") as fr:
            gold = json.load(fr)
        data.extend((data_, gold))
    return data
    
def _print_log(subtask:str, approach:str, shortcut_model_name:str, last_prompt, n_instances_processed:str):
    """
    Prints log information to a JSON file.

    Args:
        subtask (str): The subtask of the evaluation.
        approach (str): The approach used for evaluation.
        shortcut_model_name (str): The name of the model.
        last_prompt: The last prompt processed.
        n_instances_processed (str): The number of instances processed.

    Returns:
        None
    """
    log_file_path = f"../data/{subtask}/{approach}/{shortcut_model_name}/log.json"
    log = {
            "log":{
                    "subtask":subtask,
                    "approach":approach,
                    "model":shortcut_model_name,
                    "number of instances processed": n_instances_processed,
                    "last prompt":last_prompt,
                  }
          }

    with open(log_file_path, "w") as fp:
        json.dump(log, fp, indent=4)

def _prepare_finetuned_model(shortcut_model_name:str, checkpoint_path:str, trust_remote_code:bool):
    # load the original model first
    full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    if shortcut_model_name == "phi_3_mini":
        base_model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
    else:
        base_model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16).cuda()
    
    # merge fine-tuned weights with the base model
    peft_model_id = checkpoint_path
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.merge_and_unload()
    return tokenizer, model

def _process(output_file_path:str, subtask:str, approach:str, shortcut_model_name:str, is_finetuned:bool, checkpoint_path:str):
    """
    Processes the evaluation task for a specific subtask, approach, and model. Selection and generation subtasks only.

    Args:
        output_file_path (str): The path of the output foledr.
        subtask (str): The subtask of the evaluation.
        approach (str): The approach used for evaluation.
        shortcut_model_name (str): The name of the model.
        is_finetuned (bool): If the model is finetuned or not.
        checkpoint_path (str): The path of the finetuned checkpoint.

    Returns:
        None
    """
    global full_model_name2pipeline, shortcut_model_name2full_model_name

    gold_data = _get_gold_data(subtask)[0]
    n_instances_processed = 0
    json_data = []
    trust_remote_code = True

    # if the model is finetuned, the checkpoint path is needed
    if is_finetuned:
        tokenizer, model = _prepare_finetuned_model(shortcut_model_name, checkpoint_path, trust_remote_code)
        pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)
    else:
        full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
        tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token
        if shortcut_model_name == "phi_3_mini": model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
        else: model = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=trust_remote_code, torch_dtype=torch.float16).cuda()
        pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)
    
    with open(f"{output_file_path}/output.txt", "a") as fa_txt, open(f"{output_file_path}/output.json", "w") as fw_json:
        for instance in tqdm(gold_data, total=len(gold_data)):

            n_instances_processed += 1
            instance_id = instance["id"]
            
            chat_prompt = _generate_prompt(instance, subtask, approach, chat_template = True)
            prompt_template = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
            answer = pipe(prompt_template)[0]["generated_text"].replace(prompt_template, "").replace("\n", "").strip()
            
            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)

    last_prompt = prompt if shortcut_model_name == "vicuna" or shortcut_model_name == "falcon" else prompt_template
    if args.log_config:
        if is_finetuned: shortcut_model_name = f"finetuned_{shortcut_model_name}"
        _print_log(subtask, approach, shortcut_model_name, last_prompt, n_instances_processed)

def process(subtask:str, approach:str, shortcut_model_name:str, is_finetuned:bool, checkpoint_path:str):
    """
    Starts the processing for a specified subtask, approach, and model.

    Args:
        subtask (str): The subtask to be evaluated.
        approach (str): The approach used for evaluation.
        shortcut_model_name (str): The name of the model.
        is_finetuned (bool): If the model is finetuned or not.
        checkpoint_path (str): The path of the finetuned checkpoint.

    Returns:
        None
    """
    assert shortcut_model_name in supported_shortcut_model_names
    assert subtask in supported_subtasks
    assert approach in supported_approaches
    
    # we define the correct output path
    output_file_path = f"../data/{subtask}/{approach}/"
    if is_finetuned: output_file_path += f"finetuned_{shortcut_model_name}/"
    else: output_file_path += f"{shortcut_model_name}/"
    # to manage creation/deletion of folders
    if not os.path.exists(f"../data/{subtask}/"):
        os.system(f"mkdir ../data/{subtask}/")
    if not os.path.exists(f"../data/{subtask}/"):
        os.system(f"mkdir ../data/{subtask}/")
    if not os.path.exists(f"../data/{subtask}/"):
        os.system(f"mkdir ../data/{subtask}/")
    if not os.path.exists(f"../data/{subtask}/{approach}/"):
        os.system(f"mkdir ../data/{subtask}/{approach}/")
    if not os.path.exists(output_file_path):
        os.system(f"mkdir {output_file_path}")
    elif os.path.exists(f"{output_file_path}/output.txt"):
        countdown(5)
        os.system(f"rm -r {output_file_path}/*")

    _process(output_file_path, subtask, approach, shortcut_model_name, is_finetuned, checkpoint_path)

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    supported_subtasks = ["selection", "generation"]
    supported_approaches = ["zero_shot", "one_shot", "few_shot"]
    supported_shortcut_model_names = ["llama_2", "llama_3", "mistral", "falcon", "vicuna", 
                                      "tiny_llama", "stability_ai", "h2o_ai",
                                      "phi_3_small", "phi_3_mini", "gemma_2b", "gemma_9b"]
    full_model_name2pipeline = {}
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-st", type=str, help="Input the task")
    parser.add_argument("--approach", "-a", type=str, help="Input the approach")
    parser.add_argument("--shortcut_model_name", "-m", type=str, help="Input the model")
    parser.add_argument("--is_finetuned", "-f", type=bool, default=False, help="If the model we want to test is finetuned or not")
    parser.add_argument("--checkpoint_path", "-cp", type=str, default=None, help="Input the checkpoint path")
    parser.add_argument("--log_config", "-l", type=bool, default=True, help="Log the results")
    args = parser.parse_args()
    
    assert args.is_finetuned==False or args.checkpoint_path!=None
    process(args.subtask, args.approach, args.shortcut_model_name, args.is_finetuned, args.checkpoint_path)