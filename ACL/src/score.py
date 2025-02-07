from sklearn.metrics import precision_score, recall_score, f1_score
from typing import NamedTuple, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import argparse
import json
import nltk
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from utils import _get_gold_data, _generate_gold_data_vectors, _generate_disambiguated_data_vectors, _get_gold_data_vectors, _get_disambiguated_data_vectors, _write_definition_ranks, _print_scores

def _choose_definition(instance_gold, answer):
    """
    Chooses the most appropriate definition from a list of definitions for a given answer.
    
    Args:
        instance_gold (Dict): One instance of the golden data.
        answer (str): The answer to be disambiguated.
    
    Returns:
        str: The chosen definition with the highest overlap score with the answer.
    """
    id_ = instance_gold["id"]
    definitions = instance_gold["definitions"]
    
    if args.subtask == "generation": global id2vec_gold, id2vec_disambiguated_data
    if args.llm_as_judge: llm_as_judge_answer = _get_llm_as_judge_answer(instance_gold, answer)

    definition2overlap = {}
    for definition in definitions:
        if args.subtask == "generation":
            if args.llm_as_judge:
                overlap = _compute_lexical_overlap(definition, llm_as_judge_answer)
            else:   
                vec1, vec2 = id2vec_gold[id_+definition], id2vec_disambiguated_data[id_]
                overlap = _compute_semantic_overlap(vec1, vec2)
        else:
            overlap = _compute_lexical_overlap(definition, answer)
        definition2overlap[definition] = overlap
    sorted_candidates_scores = sorted(definition2overlap.items(), key=lambda item: item[1], reverse=True)
    definition_ranks_infos = {"id":id_, "gold_candidates": instance_gold["gold_definitions"], "model_response":answer, "candidates":[e[0] for e in sorted_candidates_scores], "scores":[str(round(e[1],2)) for e in sorted_candidates_scores]}
    return max(definition2overlap, key=definition2overlap.get), definition_ranks_infos

def _get_llm_as_judge_answer(instance_gold, answer):
    
    word = instance_gold["word"]
    candidate_definitions = "\n".join([f"\"{definition}\"" for definition in instance_gold["definitions"]])
    prompt = f"Given the following definitions of \"{word}\":\n\n{candidate_definitions}\n\nSelect the definition that most closely matches the definition: \"{answer}\". Do not explain your output."
    prompt_template = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    answer = pipe(prompt_template)[0]["generated_text"].replace(prompt_template, "").replace("\n", "").strip()
    return answer

def _compute_lexical_overlap(definition, answer):
    """
    Computes the overlap between two strings based on tokenized words.
    
    Args:
        definition (str): The first string to compare.
        answer (str): The second string to compare.
    
    Returns:
        float: The overlap score between the two strings, 
               computed as the ratio of the number of common tokens
               to the total number of unique tokens in both strings.
    """
    tokens1 = set(nltk.word_tokenize(definition))
    tokens2 = set(nltk.word_tokenize(answer))
    overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    return overlap

def _compute_semantic_overlap(definition:str, answer:str):
    """
    Computes the semantic overlap between two vectors representing semantic features.
    
    Args:
        definition (str): A string representation of semantic features for the definition.
        answer (str): A string representation of semantic features for the answer.
    
    Returns:
        float: The semantic overlap score between the two vectors.
    """
    definition = np.array([float(x) for x in definition.split(" ")])
    answer = np.array([float(x) for x in answer.split(" ")])
    definition = definition.reshape(1, -1)
    answer = answer.reshape(1, -1)
    similarity_matrix = cosine_similarity(definition, answer)
    return similarity_matrix[0][0]

def compute_scores(disambiguated_data_path:str):
    """
    Computes and prints evaluation scores based on disambiguated data and gold data.
    
    Args:
        disambiguated_data_path (str): The path to the disambiguated data file.
    
    Returns:
        None
    """
    print("Computing scores starting from LLM generated data...\n")
    if args.llm_as_judge:
        global pipe, tokenizer
        model_name = "google/gemma-2-9b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda()
        pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)
    if args.subtask == "generation" and args.llm_as_judge is False: # we need to instatiate embedding vectors
        global id2vec_gold, id2vec_disambiguated_data
        id2vec_gold = _get_gold_data_vectors(args.sentence_embedder)
        id2vec_disambiguated_data = _get_disambiguated_data_vectors(args.subtask, args.approach, args.shortcut_model_name, args.sentence_embedder, args.is_finetuned)

    gold_data = _get_gold_data(args.subtask)
    with open(disambiguated_data_path, "r") as json_file:
        disambiguated_data = json.load(json_file)
    assert len(gold_data) == len(disambiguated_data)

    if args.pos == "ALL": number_of_evaluation_instances = len(gold_data)
    else: number_of_evaluation_instances = len([instance_gold for instance_gold in gold_data if instance_gold["pos"] == args.pos])

    true_labels = [1 for _ in range(number_of_evaluation_instances)]
    predicted_labels = [1 for _ in range(number_of_evaluation_instances)]
    definition_ranks_list = []
    correct, wrong = 0,0
    global_idx = 0
    for instance_gold, instance_disambiguated_data in tqdm(zip(gold_data, disambiguated_data), total=len(gold_data), desc="Processing"):
        if args.pos == "ALL": pass
        elif args.pos != "ALL" and instance_gold["pos"] != args.pos: continue
        assert instance_gold["id"] == instance_disambiguated_data["instance_id"]

        answer = instance_disambiguated_data["answer"]

        # in DS setting we also need to add numbers before each gold and candidate definitions
        if args.subtask == "selection":
            for idx, definition in enumerate(instance_gold["definitions"]):
                for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                    if definition == gold_definition:
                        instance_gold["gold_definitions"][idx_] = f"{idx+1}) {instance_gold['gold_definitions'][idx_]}"
            for idx, definition in enumerate(instance_gold["definitions"]):
                instance_gold["definitions"][idx] = f"{idx+1}) {definition}"

        if answer.strip() == "": selected_definition = ""
        else: selected_definition, definition_ranks_infos = _choose_definition(instance_gold, answer); definition_ranks_list.append(definition_ranks_infos)
        
        if selected_definition in instance_gold["gold_definitions"]: correct += 1
        else: predicted_labels[global_idx] = 0; wrong += 1

        global_idx += 1
    assert correct+wrong == number_of_evaluation_instances
    
    # we create definition_ranks file at the end (in this way the next time we save time in doing the same computations)
    definition_ranks_path = f"{disambiguated_data_path[:-11]}/definition_ranks.json"
    if args.pos == "ALL": _write_definition_ranks(definition_ranks_path, definition_ranks_list, args.llm_as_judge)

    # we finally print the scores
    _print_scores(true_labels, predicted_labels, number_of_evaluation_instances, correct, wrong)

def compute_scores_from_file(definition_ranks_data):
    print("Computing scores starting from definition_ranks.json file...\n")
    gold_data = _get_gold_data(args.subtask)
    assert len(gold_data) == len(definition_ranks_data)

    if args.pos == "ALL": number_of_evaluation_instances = len(gold_data)
    else: number_of_evaluation_instances = len([instance_gold for instance_gold in gold_data if instance_gold["pos"] == args.pos])
    true_labels = [1 for _ in range(number_of_evaluation_instances)]
    predicted_labels = [1 for _ in range(number_of_evaluation_instances)]
    correct, wrong = 0,0
    global_idx = 0
    for instance_gold, instance_definition_ranks in tqdm(zip(gold_data, definition_ranks_data), total=len(gold_data), desc="Processing"):
        if args.pos == "ALL": pass
        elif args.pos != "ALL" and instance_gold["pos"] != args.pos: continue
        assert instance_gold["id"] == instance_definition_ranks["id"]

        # in DS setting we also need to add numbers before each gold and candidate definitions
        if args.subtask == "selection":
            for idx, definition in enumerate(instance_gold["definitions"]):
                for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                    if definition == gold_definition:
                        instance_gold["gold_definitions"][idx_] = f"{idx+1}) {instance_gold['gold_definitions'][idx_]}"
            for idx, definition in enumerate(instance_gold["definitions"]):
                instance_gold["definitions"][idx] = f"{idx+1}) {definition}"

        choosen_candidate = ""
        if args.llm_as_judge == True:
            for candidate in instance_definition_ranks["candidates"]:
                if candidate[:4] == "****": choosen_candidate  = candidate[5:-5] ; break
        else:
            if args.subtask == "generation" and instance_definition_ranks["candidates"][0][:4] == "****":
                choosen_candidate  = instance_definition_ranks["candidates"][0][5:-5]
            else: choosen_candidate = instance_definition_ranks["candidates"][0]
        
        if choosen_candidate in instance_gold["gold_definitions"]: correct += 1
        else: predicted_labels[global_idx] = 0; wrong += 1

        global_idx += 1
    assert correct+wrong == number_of_evaluation_instances

    # we finally print the scores
    _print_scores(true_labels, predicted_labels, number_of_evaluation_instances, correct, wrong)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--subtask', '-s', type=str, help='Input the task')
    parser.add_argument('--approach', '-a', type=str, help='Input the approach')
    parser.add_argument('--shortcut_model_name', '-sm', type=str, help='Input the model')
    parser.add_argument("--is_finetuned", "-f", type=bool, default=False, help="If the model we want to test is finetuned or not")
    parser.add_argument('--pos', '-p', type=str, help='Input the part of speech')
    parser.add_argument('--sentence_embedder', '-se', type=str, default=None, help='Input the sentence embedder')
    parser.add_argument('--llm_as_judge', '-j', type=bool, default=False, help='If we are using llm-as-a-judge for the generation setting')
    args = parser.parse_args()

    assert args.subtask in ["generation", "selection"]
    assert args.approach in ["zero_shot", "one_shot", "few_shot"]
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
    assert args.pos in ["NOUN", "ADJ", "VERB", "ADV", "ALL"]
    assert args.llm_as_judge is False or args.subtask == "generation"

    # particular handling if we are going to score a FINETUNED model
    disambiguated_data_path = f"../data/{args.subtask}/{args.approach}/"
    model_name = f"finetuned_{args.shortcut_model_name}" if args.is_finetuned else f"{args.shortcut_model_name}"
    definition_ranks_path = f"{disambiguated_data_path}{model_name}/definition_ranks.json"
    disambiguated_data_path += f"{model_name}/output.json"

    # when we want to perform Definition Generation using cosine similarity (we need to create vectors)
    # and the definition_ranks file has not been created yet
    if args.subtask == "generation" and args.llm_as_judge is False and not os.path.isfile(definition_ranks_path):
        assert args.sentence_embedder in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
        len_gold = len(_get_gold_data(args.subtask))
        _generate_gold_data_vectors(args.subtask, args.sentence_embedder)
        _generate_disambiguated_data_vectors(args.subtask, args.approach, args.shortcut_model_name, args.sentence_embedder, args.is_finetuned, disambiguated_data_path, len_gold)
    
    # different scenarios: 
    # 1) if definition_ranks file already exists
    if os.path.isfile(definition_ranks_path):
        with open(definition_ranks_path, "r") as json_file:
            definition_ranks_data = json.load(json_file)
        # if we want to run llm_as_judge and definition_ranks file already exists 
        # we need to check if it has already been run or not
        # 1.1)
        if args.llm_as_judge == True:
            has_llm_as_judge_been_run = False
            for candidate in definition_ranks_data[-1]["candidates"]:
                if candidate[:4]=="****": has_llm_as_judge_been_run = True ; break
            # 1.1.1) if already been run, we compute the scores from definition_ranks file
            if has_llm_as_judge_been_run: compute_scores_from_file(definition_ranks_data)
            # 1.1.2) if not, we need to run it and populate definition_ranks with "****"
            else: compute_scores(disambiguated_data_path)
        # 1.2)
        else: compute_scores_from_file(definition_ranks_data)
    # 2) at the end of the process definition_ranks will be created
    else: compute_scores(disambiguated_data_path)