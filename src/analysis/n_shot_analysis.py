from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from analysis_variables import shortcut_model_name2full_model_name, prompts
from tqdm import tqdm
import warnings
import argparse
import time
import json
import torch
import os
import nltk
import numpy as np

################
# DISAMBIGUATE #
################

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

def _generate_prompt(instance:dict, ambiguity:str, most_frequent:str, approach:str):

    word = instance["word"]
    text = instance["text"].replace(" ,", ",").replace(" .", ".")
    candidate_definitions = "\n".join([f"{idx}) {x}" for idx, x in enumerate(instance["definitions"])])
    
    prompt = prompts["n_shot_analysis"][ambiguity][most_frequent][approach].format(
            word=word,
            text=text,
            candidate_definitions=candidate_definitions)

    return prompt

def _get_gold_data():
    """
    Loads gold data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    data_path = "../../data/evaluation/ALL_preprocessed.json"
    with open(data_path, "r") as json_file:
        gold_data = json.load(json_file)
    return gold_data

def disambiguate(ambiguity:str, most_frequent:str, approach:str, shortcut_model_name:str):
    assert shortcut_model_name in supported_shortcut_model_names
    assert ambiguity in supported_ambiguity
    assert approach in supported_approaches
    assert most_frequent in supported_mfs
    global shortcut_model_name2full_model_name

    gold_data = _get_gold_data()
    output_file_path = f"../../data/analysis/n_shot_analysis/{ambiguity}/{most_frequent}/{approach}/{shortcut_model_name}"
    n_instances_processed = 0
    json_data = []

    # to manage creation/deletion of folders
    if not os.path.exists(f"../../data/analysis/"):
        os.system(f"mkdir ../../data/analysis/")
    if not os.path.exists(f"../../data/analysis/n_shot_analysis/"):
        os.system(f"mkdir ../../data/analysis/n_shot_analysis/")
    if not os.path.exists(f"../../data/analysis/n_shot_analysis/{ambiguity}/"):
        os.system(f"mkdir ../../data/analysis/n_shot_analysis/{ambiguity}/")
    if not os.path.exists(f"../../data/analysis/n_shot_analysis/{ambiguity}/{most_frequent}/"):
        os.system(f"mkdir ../../data/analysis/n_shot_analysis/{ambiguity}/{most_frequent}/")
    if not os.path.exists(f"../../data/analysis/n_shot_analysis/{ambiguity}/{most_frequent}/{approach}/"):
        os.system(f"mkdir ../../data/analysis/n_shot_analysis/{ambiguity}/{most_frequent}/{approach}/")
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
            prompt = _generate_prompt(instance, ambiguity, most_frequent, approach)

            answer = pipe(prompt)[0]["generated_text"].replace(prompt, "").replace("\n", "").strip()

            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)

#########
# SCORE #
#########

def _choose_definition(instance_gold, answer):
    id_ = instance_gold["id"]
    definitions = instance_gold["definitions"]
    definition2overlap = {}
    for definition in definitions:
        overlap = _compute_lexical_overlap(definition, answer)
        definition2overlap[definition] = overlap
    return max(definition2overlap, key=definition2overlap.get)

def _compute_lexical_overlap(definition, answer):
    tokens1 = set(nltk.word_tokenize(definition))
    tokens2 = set(nltk.word_tokenize(answer))
    overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    return overlap

def _get_disambiguated_data(disambiguated_data_path:str):
    with open(disambiguated_data_path, "r") as json_file:
        disambiguated_data = json.load(json_file)
    return disambiguated_data

def compute_scores(disambiguated_data_path:str):

    gold_data = _get_gold_data()
    disambiguated_data = _get_disambiguated_data(disambiguated_data_path)
    assert len(gold_data) == len(disambiguated_data)
    
    true_labels = [1 for _ in range(len(gold_data))]
    predicted_labels = [1 for _ in range(len(gold_data))]

    correct_most_frequent, not_correct_most_frequent, correct_not_most_frequent, not_correct_not_most_frequent = 0, 0, 0, 0
    correct, wrong = 0,0
    global_idx = 0

    for instance_gold, instance_disambiguated_data in zip(gold_data, disambiguated_data):
        assert instance_gold["id"] == instance_disambiguated_data["instance_id"]

        answer = instance_disambiguated_data["answer"]

        # adds n) before each gold definition
        for idx, definition in enumerate(instance_gold["definitions"]):
            for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                if definition == gold_definition:
                    instance_gold["gold_definitions"][idx_] = f"{idx}) {instance_gold['gold_definitions'][idx_]}"
        # adds n) before all candidate definitions
        for idx, definition in enumerate(instance_gold["definitions"]):
            instance_gold["definitions"][idx] = f"{idx}) {definition}"
            
        def2idx = {d:i for i,d in enumerate(instance_gold["definitions"])}

        if answer.strip() == "": selected_definition = ""
        else: selected_definition = _choose_definition(instance_gold, answer)
        
        if selected_definition in instance_gold["gold_definitions"]:
            if def2idx[selected_definition]==0: correct_most_frequent+=1
            else: correct_not_most_frequent+=1
            correct += 1
        else:
            predicted_labels[global_idx] = 0
            if def2idx[selected_definition]==0: not_correct_most_frequent+=1
            else: not_correct_not_most_frequent+=1
            wrong += 1

        global_idx += 1
    assert correct+wrong == len(gold_data)
    
    perc_mfs_predicted = round(((correct_most_frequent+not_correct_most_frequent)/500)*100,2)
    perc_not_mfs_correctly_predicted = round( ((correct_not_most_frequent)/(correct_not_most_frequent+not_correct_not_most_frequent))*100 , 2)
    acc = round((correct/len(gold_data))*100,2)
    return perc_mfs_predicted, perc_not_mfs_correctly_predicted, acc

def score(approach:str, shortcut_model_name:str):
    # MFS analysis
    ambiguity_level = "6"
    most_frequent_list = ["mfs", "not_mfs"]
    mfs_ris = []
    for most_frequent in most_frequent_list:
        disambiguated_data_path = f"../../data/analysis/n_shot_analysis/{ambiguity_level}/{most_frequent}/{args.approach}/{args.shortcut_model_name}/output.json"
        perc_mfs_predicted, perc_not_mfs_correctly_predicted, acc = compute_scores(disambiguated_data_path)
        mfs_ris.append([perc_mfs_predicted, perc_not_mfs_correctly_predicted, acc])
    print("# MFS analysis")
    table_values=[["", "% mfs predicted", "% not mfs predicted correctly", "f1-score"],
                  ["MFS", str(mfs_ris[0][0])+"%", str(mfs_ris[0][1])+"%", str(mfs_ris[0][2])],
                  ["not MFS", str(mfs_ris[1][0])+"%", str(mfs_ris[1][1])+"%", str(mfs_ris[1][2])]]
    col_widths = [max(len(str(cell)) for cell in column) for column in zip(*table_values)]
    for row in table_values:
        print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))
    print(f"The loss is about -{mfs_ris[0][2]-mfs_ris[1][2]}%.")
    print()
    
    # AMBIGUITY analysis
    ambiguity_list = ["1", "3", "6", "10", "16"]
    ambiguity_ris = []
    for most_frequent in most_frequent_list:
        l = []
        for ambiguity_level in ambiguity_list:
            disambiguated_data_path = f"data/analysis/n_shot_analysis/{ambiguity_level}/{most_frequent}/{args.approach}/{args.shortcut_model_name}/output.json"
            _, _, acc = compute_scores(disambiguated_data_path)
            l.append(acc)
        std = np.asarray(l).std()
        l.append(std)
        ambiguity_ris.append(l)
    print("# AMBIGUITY analysis")
    table_values=[["", "#1", "#3", "#6", "#10", "#16", "std"],
                  ["MFS", str(ambiguity_ris[0][0]), str(ambiguity_ris[0][1]), str(ambiguity_ris[0][2]), str(ambiguity_ris[0][3]), str(ambiguity_ris[0][4]), str(ambiguity_ris[0][5])],
                  ["MFS", str(ambiguity_ris[1][0]), str(ambiguity_ris[1][1]), str(ambiguity_ris[1][2]), str(ambiguity_ris[1][3]), str(ambiguity_ris[1][4]), str(ambiguity_ris[1][5])]]
    col_widths = [max(len(str(cell)) for cell in column) for column in zip(*table_values)]
    for row in table_values:
        print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))
    print()

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    supported_mode = ["disambiguate", "score"]
    supported_ambiguity = ["1", "3", "6", "10", "16"]
    supported_mfs = ["mfs", "not_mfs"]
    supported_approaches = ["one_shot", "few_shot"]
    supported_shortcut_model_names = ["llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2", "falcon-7b-instruct", "vicuna-7b-v1.5",
                                      "microsoft-phi-1_5", "TinyLlama-TinyLlama-1.1B-Chat-v1.0", "stabilityai-stablelm-2-1_6b-chat", "h2oai-h2o-danube2-1.8b-chat",
                                      "microsoft-phi-2", "microsoft-phi-3-mini-128k-instruct", "meta-llama-Meta-Llama-3-8B",
                                      "openlm-research-open_llama_3b_v2", "openlm-research-open_llama_7b_v2"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, help="Input the mode (disambiguate or score)")
    parser.add_argument("--ambiguity", "-am", type=str, help="Input the ambiguity level")
    parser.add_argument("--most_frequent", "-mf", type=str, help="Input the most frequent level")
    parser.add_argument("--approach", "-a", type=str, help="Input the approach")
    parser.add_argument("--shortcut_model_name", "-mn", type=str, help="Input the model")
    args = parser.parse_args()
    
    assert args.ambiguity!="1" or args.most_frequent=="mfs"
    if args.mode == "disambiguate":
        disambiguate(args.ambiguity, args.most_frequent, args.approach, args.shortcut_model_name)
    else:
        score(args.approach, args.shortcut_model_name)