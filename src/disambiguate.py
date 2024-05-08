from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from variables import shortcut_model_name2full_model_name, prompts
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

def _generate_prompt(instance:dict, subtask:str, prompt_type:str, prompt_addition:str, approach:str):
    """
    Generates a prompt based on the instance, subtask, prompt_type, prompt_addition and approach.

    Args:
        instance (dict): The instance containing information about the task.
        subtask (str): The subtask of the instance, e.g., "selection", "generation", or "wic".
        prompt_type (str): The prompt type used.
        prompt_addition (str): The prompt addition technique added to the nominal prompt.
        approach (str): The approach used for generating the prompt.

    Returns:
        str or None: The generated prompt if applicable, otherwise None.
    """
    if subtask in ["selection", "generation"]:

        word = instance["word"]
        text = instance["text"].replace(" ,", ",").replace(" .", ".")
        
        if subtask == "selection" and prompt_type == "v3":
            candidate_definitions = "\n".join([f"- [{sense_key}]: {definition};" for sense_key, definition in list(zip(instance["candidates"], instance["definitions"]))])
        else:
            candidate_definitions = "\n".join([f"{idx}) {x}" for idx, x in enumerate(instance["definitions"])])
        
        prompt = prompts[subtask][prompt_type][prompt_addition][approach].format(
                word=word,
                text=text,
                candidate_definitions=candidate_definitions)

    elif subtask == "wic":
        "To be implemented"
        pass

    return prompt

def _get_gold_data():
    """
    Loads gold data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    data_path = "../data/evaluation/ALL_preprocessed.json"
    with open(data_path, "r") as json_file:
        gold_data = json.load(json_file)
    return gold_data
    
def _print_log(subtask:str, prompt_type:str, prompt_addition:str, approach:str, shortcut_model_name:str, last_prompt, n_instances_processed:str):
    """
    Prints log information to a JSON file.

    Args:
        subtask (str): The subtask of the evaluation.
        prompt_type (str): The prompt type used.
        prompt_addition (str): The prompt addition technique added to the nominal prompt.
        approach (str): The approach used for evaluation.
        shortcut_model_name (str): The name of the model.
        last_prompt: The last prompt processed.
        n_instances_processed (str): The number of instances processed.

    Returns:
        None
    """
    log_file_path = f"../data/{subtask}/{prompt_type}/{prompt_addition}/{approach}/{shortcut_model_name}/log.json"
    log = {
            "log":{
                    "subtask":subtask,
                    "prompt_type":prompt_type,
                    "prompt_addition":prompt_addition,
                    "approach":approach,
                    "model":shortcut_model_name,
                    "number of instances processed": n_instances_processed,
                    "last prompt":last_prompt,
                  }
          }

    with open(log_file_path, "w") as fp:
        json.dump(log, fp, indent=4)

def _process(subtask:str, prompt_type:str, prompt_addition:str, approach:str, shortcut_model_name:str):
    """
    Processes the evaluation task for a specific subtask, approach, and model. Selection and generation subtasks only.

    Args:
        subtask (str): The subtask of the evaluation.
        prompt_type (str): The prompt type used.
        prompt_addition (str): The prompt addition technique added to the nominal prompt.
        approach (str): The approach used for evaluation.
        shortcut_model_name (str): The name of the model.

    Returns:
        None
    """
    global full_model_name2pipeline, shortcut_model_name2full_model_name

    gold_data = _get_gold_data()
    output_file_path = f"../data/{subtask}/{prompt_type}/{prompt_addition}/{approach}/{shortcut_model_name}/"
    n_instances_processed = 0
    json_data = []

    # to manage creation/deletion of folders
    if not os.path.exists(f"../data/{subtask}/{prompt_type}/"):
        os.system(f"mkdir ../data/{subtask}/{prompt_type}/")
    if not os.path.exists(f"../data/{subtask}/{prompt_type}/{prompt_addition}/"):
        os.system(f"mkdir ../data/{subtask}/{prompt_type}/{prompt_addition}/")
    if not os.path.exists(f"../data/{subtask}/{prompt_type}/{prompt_addition}/{approach}/"):
        os.system(f"mkdir ../data/{subtask}/{prompt_type}/{prompt_addition}/{approach}/")
    if not os.path.exists(output_file_path):
        os.system(f"mkdir {output_file_path}")
    elif os.path.exists(f"{output_file_path}/output.txt"):
        countdown(5)
        os.system(f"rm -r {output_file_path}/*")

    full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline("text-generation", model=full_model_name, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)

    with open(f"{output_file_path}/output.txt", "a") as fa_txt, open(f"{output_file_path}/output.json", "w") as fw_json:
        for instance in tqdm(gold_data, total=len(gold_data)):

            n_instances_processed += 1
            instance_id = instance["id"]
            prompt = _generate_prompt(instance, subtask, prompt_type, prompt_addition, approach)

            answer = pipe(prompt)[0]["generated_text"].replace(prompt, "").replace("\n", "").strip()

            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)

    last_prompt= prompt
    if args.log_config:
        _print_log(subtask, prompt_type, prompt_addition, approach, shortcut_model_name, last_prompt, n_instances_processed)

def _process_wic(subtask:str, prompt_type:str, prompt_addition:str, approach:str, shortcut_model_name:str):
    "To be implemented"
    pass
    
def process(subtask:str, prompt_type:str, prompt_addition:str, approach:str, shortcut_model_name:str):
    """
    Starts the processing for a specified subtask, approach, and model.

    Args:
        subtask (str): The subtask to be evaluated.
        prompt_type (str): The prompt type used.
        prompt_addition (str): The prompt addition technique added to the nominal prompt.
        approach (str): The approach used for evaluation.
        shortcut_model_name (str): The name of the model.

    Returns:
        None
    """
    assert shortcut_model_name in supported_shortcut_model_names
    assert subtask in supported_subtasks
    assert approach in supported_approaches
    assert prompt_type in supported_prompt_types
    assert prompt_addition in supported_prompt_additions

    if subtask in ["selection", "generation"]:
        _process(subtask, prompt_type, prompt_addition, approach, shortcut_model_name)

    elif subtask == "wic":
        _process_wic(subtask, prompt_type, prompt_addition, approach, shortcut_model_name)

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    supported_subtasks = ["selection", "generation", "wic"]
    supported_prompt_types = ["v1", "v1.1", "v1.2", "v2", "v2.1", "v2.2", "v3", "v3.1", "v3.2"]
    supported_prompt_additions = ["no_additions", "cot", "reflective", "cognitive", "emotion"]
    supported_approaches = ["zero_shot", "one_shot", "few_shot"]
    supported_shortcut_model_names = ["llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2", "falcon-7b-instruct", "vicuna-7b-v1.5", "TowerInstruct-7B-v0.1", 
                                      "microsoft-phi-1_5", "TinyLlama-TinyLlama-1.1B-Chat-v1.0", "stabilityai-stablelm-2-1_6b-chat", "h2oai-h2o-danube2-1.8b-chat"]
    full_model_name2pipeline = {}
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-st", type=str, help="Input the task")
    parser.add_argument("--prompt_type", "-pt", type=str, help="Input the prompt type")
    parser.add_argument("--prompt_addition", "-pa", type=str, help="Input the prompt addition")
    parser.add_argument("--approach", "-a", type=str, help="Input the approach")
    parser.add_argument("--shortcut_model_name", "-m", type=str, help="Input the model")
    parser.add_argument("--log_config", "-l", type=bool, default=True, help="Log the results")
    args = parser.parse_args()
    process(args.subtask, args.prompt_type, args.prompt_addition, args.approach, args.shortcut_model_name)
