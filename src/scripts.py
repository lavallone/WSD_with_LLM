from tqdm import tqdm
import json
import os
import random
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.contingency_tables import mcnemar 
import seaborn as sns
import matplotlib.pyplot as plt


def mc_nemar_test(subtask, approach):

    # the idea is to generate two sets of list for both models (consec and gpt): 
    # one containing the set of correct instances and one containing the wrong ones
    consec_gold_path = '../data/evaluation/ALLamended/xml/ALLamended.gold.key.txt'
    consec_gold_data = {}
    with open(consec_gold_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            consec_gold_data[parts[0]] = parts[1:]
    
    consec_preds_path = '../data/evaluation/ALLamended/xml/predictions.consec.key.txt'
    consec_preds_data = {}
    with open(consec_preds_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            consec_preds_data[parts[0]] = parts[1:]

    consec_corrects, consec_wrongs = [], []
    for k,v in consec_preds_data.items():
        if v[0] in consec_gold_data[k]: consec_corrects.append(k)
        else: consec_wrongs.append(k)
    
    # now we pass to gpt-4o model
    gpt_gold_path = "../data/evaluation/ALLamended_preprocessed.json"
    gpt_preds_path = f"../data/{subtask}/{approach}/gpt/definition_ranks.json"
    with open(gpt_gold_path, "r") as json_file:
        gold_data = json.load(json_file)
    with open(gpt_preds_path, "r") as json_file:
        disambiguated_data = json.load(json_file)
    assert len(gold_data) == len(disambiguated_data)

    number_of_evaluation_instances = len(gold_data)
    gpt_corrects, gpt_wrongs = [], []
    for instance_gold, instance_disambiguated_data in tqdm(zip(gold_data, disambiguated_data), total=len(gold_data), desc="Processing"):
        
        assert instance_gold["id"] == instance_disambiguated_data["id"]

        if subtask == "selection":
            # adds n) before each gold definition
            for idx, definition in enumerate(instance_gold["definitions"]):
                for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                    if definition == gold_definition:
                        instance_gold["gold_definitions"][idx_] = f"{idx+1}) {instance_gold['gold_definitions'][idx_]}"
            # adds n) before all candidate definitions
            for idx, definition in enumerate(instance_gold["definitions"]):
                instance_gold["definitions"][idx] = f"{idx+1}) {definition}"

        choosen_candidate = ""
        if subtask == "generation" and instance_disambiguated_data["candidates"][0][:4] == "****":
            choosen_candidate  = instance_disambiguated_data["candidates"][0][5:-5]
        else: choosen_candidate = instance_disambiguated_data["candidates"][0]
        
        if choosen_candidate in instance_gold["gold_definitions"]: gpt_corrects.append(instance_gold["id"])
        else: gpt_wrongs.append(instance_gold["id"])

    # once we have the four lists we can compute the McNemar's test for statistical significativity
    correct_gpt_set = set(gpt_corrects)
    wrong_gpt_set = set(gpt_wrongs)
    correct_consec_set = set(consec_corrects)
    wrong_consec_set = set(consec_wrongs)
    n00 = len(wrong_gpt_set.intersection(wrong_consec_set)) # misclassified by both
    n01 = len(wrong_gpt_set.difference(wrong_consec_set)) # misclassified by GPT but not ConSeC
    n10 = len(wrong_consec_set.difference(wrong_gpt_set)) # misclassified by ConSeC but not GPT
    n11 = len(correct_gpt_set.intersection(correct_consec_set)) # correctly classified by both
    print(f"\nn00 (Misclassified by both): {n00}")
    print(f"n01 (Misclassified by GPT, but not ConSeC): {n01}")
    print(f"n10 (Misclassified by ConSeC, but not GPT): {n10}")
    print(f"n11 (Correctly classified by both): {n11}\n")

    data = [[n00, n01], [n10, n11]]
    threshold = 6.635 # this value has been found on the Chi-square table
    significance_value = 0.01 # we want a p-value < 0.01
    # McNemar's Test with the continuity correction
    test = mcnemar(data, exact=False, correction=True)
    print(test)
    print()
    if test.pvalue < significance_value: print("Reject Null hypotesis")
    else: print("Fail to reject Null hypotesis")


def manual_analysis(approach):
    
    ## CONSEC
    # we first need to collect the instance ids correctly and wrongly classified by ConSec
    consec_gold_path = '../data/evaluation/ALLamended/xml/ALLamended.gold.key.txt'
    consec_gold_data = {}
    with open(consec_gold_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            consec_gold_data[parts[0]] = parts[1:]
    consec_preds_path = '../data/evaluation/ALLamended/xml/predictions.consec.key.txt'
    consec_preds_data = {}
    with open(consec_preds_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            consec_preds_data[parts[0]] = parts[1:]
    consec_corrects, consec_wrongs = [], []
    for k,v in consec_preds_data.items():
        if v[0] in consec_gold_data[k]: consec_corrects.append(k)
        else: consec_wrongs.append(k)

    ## GPT-4o
    # we then need to do the same thing for gpt-4o in the generation setting with gpt_as_judge
    gpt_gold_path = "../data/evaluation/ALLamended_preprocessed.json"
    gpt_preds_path = f"../data/generation/{approach}/gpt/definition_ranks.json"
    with open(gpt_gold_path, "r") as json_file:
        gold_data = json.load(json_file)
    with open(gpt_preds_path, "r") as json_file:
        disambiguated_data = json.load(json_file)
    assert len(gold_data) == len(disambiguated_data)

    gpt_wrongs, gpt_corrects = [],[]
    for instance_gold, instance_disambiguated_data in tqdm(zip(gold_data, disambiguated_data), total=len(gold_data), desc="Processing"):
        
        assert instance_gold["id"] == instance_disambiguated_data["id"]
        # candidate choosen by gpt_as_judge
        choosen_candidate = ""
        for candidate in instance_disambiguated_data["candidates"]:
            if candidate[:4] == "****": choosen_candidate = candidate[5:-5] ; break
        if choosen_candidate not in instance_gold["gold_definitions"]: gpt_wrongs.append(instance_gold["id"])
        else: gpt_corrects.append(instance_gold["id"])

    # we now need to create 3 sets of 100 instances each: 
    # 1) instances misclassifed by both models;
    # 2) instances misclassified only by gpt-4o;
    # 3) instances misclassified only by ConSec;
    wrong_gpt_set = set(gpt_wrongs)
    correct_gpt_set = set(gpt_corrects)
    wrong_consec_set = set(consec_wrongs)
    correct_consec_set = set(consec_corrects)

    misclassified_by_both = wrong_gpt_set.intersection(wrong_consec_set) # misclassified by both
    misclassified_only_by_gpt = wrong_gpt_set.intersection(correct_consec_set) # misclassified by GPT but not ConSeC
    misclassified_only_by_consec = wrong_consec_set.intersection(correct_gpt_set) # misclassified by ConSeC but not GPT
    
    misclassified_by_both = random.sample(list(misclassified_by_both), 100)
    misclassified_only_by_gpt = random.sample(list(misclassified_only_by_gpt), 100)
    misclassified_only_by_consec = random.sample(list(misclassified_only_by_consec), 100)
    ris = {"misclassified_by_both" : misclassified_by_both, "misclassified_only_by_gpt" : misclassified_only_by_gpt, "misclassified_only_by_consec" : misclassified_only_by_consec}
    
    # we finally need to retrieve the choosen definitions by the two models and create the file for manual analysis
    # we first create the skeleton of the file
    for k,v in ris.items():
        for i in range(len(v)):
            for e in gold_data:
                if e["id"] == v[i]:
                    v[i] = {"id": e["id"], "text": e["text"], "lemma": e["lemma"], "pos": e["pos"],
                            "gold_definitions": e["gold_definitions"], "definitions": e["definitions"],
                            "gpt_response" : "",
                            "gpt_candidate": "", 
                            "consec_candidate": ""} ; break
    # gpt-4o
    for k,v in ris.items():
        for i in range(len(v)):
            for instance_disambiguated_data in disambiguated_data:
                if instance_disambiguated_data["id"] == v[i]["id"]:
                    v[i]["gpt_response"] = instance_disambiguated_data["model_response"]
                    for candidate in instance_disambiguated_data["candidates"]:
                        if candidate[:4] == "****":
                            v[i]["gpt_candidate"] = candidate[5:-5] ; break

    # consec
    # id --> synset --> babelnet -->  ALLamended_preprocessed.json
    consec_id2pred = {}
    with open("../data/evaluation/ALLamended/xml/predictions.consec.key.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            consec_id2pred[parts[0]] = parts[1]
    # with open("../data/evaluation/ALLamended/babelnet/synset2babel.json", "r") as json_file:
    #     synset2babel = json.load(json_file)
    for k,v in ris.items():
        for i in range(len(v)):
            for e in gold_data:
                if e["id"] == v[i]["id"]:
                    for idx,candidate in enumerate(e["candidates"]):
                        #if candidate == synset2babel[ consec_id2pred[e["id"]] ]:
                        if candidate == consec_id2pred[e["id"]]:
                            v[i]["consec_candidate"] = e["definitions"][idx]
    # we finally save the file for manual analysis
    if not os.path.exists("../outputs/"):
        os.system("mkdir ../outputs/")
    with open("../outputs/consec_vs_gpt.json", "w") as json_file:
        json.dump(ris, json_file, indent=4)


def compute_task_correlations():

    # load data
    data = pd.read_csv("../data/evaluation/llm_performances.csv")

    wsd_tasks = ["DS", "DG"]
    benchmarks = ["ARC_Challenge", "ARC_Easy", "BoolQ", "Hellaswag", "MMLU_humanities", "MMLU_other", "MMLU_social_sciences", "MMLU_stem", "PIQA", "SciQ", "TruthfulQA_mc1", "TruthfulQA_mc2", "WinoGrande"]

    # compute Pearson correlations
    results = []
    for wsd in wsd_tasks:
        for bench in benchmarks:
            r,_ = pearsonr(data[wsd], data[bench])
            results.append({"WSD": wsd, "LLM Benchmarks": bench, "r": r})

    # show results
    results_df = pd.DataFrame(results)
    heatmap_data = results_df.pivot(index="LLM Benchmarks", columns="WSD", values="r")
    heatmap_data = heatmap_data[['DS', 'DG']]

    plt.figure(figsize=(4, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="Blues", vmin=-1, vmax=1)
    plt.title("Correlation between WSD Tasks and LLM common Benchmarks")
    if not os.path.exists("../outputs/"):
        os.system("mkdir ../outputs/")
    plt.savefig("../outputs/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    # if you want to perform the McNemar's test with Consec performances
    #mc_nemar_test("selection", "zero_shot")

    # if you want to produce a file for MANUAL ANALYSIS
    #manual_analysis("zero_shot")

    # if you want to compute linear correlations between LLM tasks
    compute_task_correlations()