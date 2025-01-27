from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import os
import time


## UTILS

def _get_gold_data(subtask:str):
    """
    Loads gold data from a JSON file.

    Returns:
        dict: A dictionary containing the loaded gold data.
    """
    gold_data_path = "../data/evaluation/ALLamended/ALLamended_preprocessed.json"
    with open(gold_data_path, "r") as json_file:
        gold_data = json.load(json_file)
    return gold_data

# disambiguate.py
###########################################################################################################

def _create_folder(subtask, approach, shortcut_model_name, is_finetuned):
    
    def _countdown(t):
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
    
    # we define the correct output path
    output_file_path = f"../data/{subtask}/{approach}/"
    if is_finetuned: output_file_path += f"finetuned_{shortcut_model_name}/"
    else: output_file_path += f"{shortcut_model_name}/"
    # to manage creation/deletion of folders
    if not os.path.exists(f"../data/{subtask}/"):
        os.system(f"mkdir ../data/{subtask}/")
    if not os.path.exists(f"../data/{subtask}/{approach}/"):
        os.system(f"mkdir ../data/{subtask}/{approach}/")
    if not os.path.exists(output_file_path):
        os.system(f"mkdir {output_file_path}")
    elif os.path.exists(f"{output_file_path}/output.txt"):
        _countdown(5)
        os.system(f"rm -r {output_file_path}/*")
    return output_file_path

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

###########################################################################################################

# score.py
###########################################################################################################

def _generate_gold_data_vectors(subtask:str):
    """
    Generates sentence embeddings for gold data and saves them to a file.

    Args:
        subtask (str)

    Returns:
        None
    """
    gold_vector_file_path = f"../data/evaluation/vectors/{args.sentence_embedder}_id2vec.tsv"
    if os.path.exists(gold_vector_file_path):
        print("Gold vectors already exist")
        return None
    gold_vector_folder_path = f"../data/evaluation/vectors"
    if not os.path.exists(gold_vector_folder_path):
        os.makedirs(gold_vector_folder_path)

    print("Generating vectors from gold data")
    sentence_embedder = SentenceTransformer(f'sentence-transformers/{args.sentence_embedder}')
    data = _get_gold_data(subtask)[0]

    gold_vector_file_path = f"../data/evaluation/vectors/{args.sentence_embedder}_id2vec.tsv"
    with open(gold_vector_file_path, "w") as fw:
        for el in tqdm(data, total=len(data)):
            id_ = el["id"]
            definitions = el["definitions"]
            for definition in definitions:
                vec = sentence_embedder.encode(definition).tolist()
                vec = " ".join([str(x) for x in vec])
                fw.write(f"{id_}\t{definition}\t{vec}\n")

def _generate_disambiguated_data_vectors(disambiguated_data_path:str, len_gold:int, is_finetuned:bool):
    """
    Generates sentence embeddings for disambiguated data and saves them to a file.

    Args:
        disambiguated_data_path (str): the path to the data disambiguated file 
        len_gold (int): number of instances in the gold file
        is_finetuned (bool): if the model is finetuned or not
    Returns:
        None
    """
    vector_file_path = f"../data/{args.subtask}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{args.subtask}/{args.approach}/{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv"
    if os.path.exists(vector_file_path):
        with open(vector_file_path, "r") as fr:
            if len(fr.readlines()) != len_gold:
                print("Missing vectors")
                exit()
            print("Disambiguated data vectors already exist")
            return None
    vector_folder_path = f"../data/{args.subtask}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors" if is_finetuned else f"../data/{args.subtask}/{args.approach}/{args.shortcut_model_name}/vectors"
    if not os.path.exists(vector_folder_path):
        os.makedirs(vector_folder_path)          

    print("Generating vectors from:", disambiguated_data_path)
    sentence_embedder = SentenceTransformer(f'sentence-transformers/{args.sentence_embedder}')

    with open(disambiguated_data_path) as fr:
        data = json.load(fr)

    vector_file_path = f"../data/{args.subtask}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{args.subtask}/{args.approach}/{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv"
    id2vec_dd = {}
    with open(vector_file_path, "w") as fw:
        for el in tqdm(data, total=len(data)):
            id_ = el["instance_id"]
            answer = el["answer"]
            vec = sentence_embedder.encode(answer)
            id2vec_dd[id_] = " ".join([str(x) for x in vec.tolist()])
            vec = " ".join([str(x) for x in vec])
            fw.write(f"{id_}\t{vec}\n")

def _get_disambiguated_data_vectors(is_finetuned:bool):
    """
    Retrieves the disambiguated data sentence embeddings from a file and returns them as a dictionary.

    Args:
        is_finetuned (bool): if the model is finetuned or not

    Returns:
        dict: A dictionary mapping instance IDs to their corresponding sentence embeddings.
    """
    id2vec_disambiguated_data = {}

    vector_file_path = f"../data/{args.subtask}/{args.approach}/finetuned_{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{args.subtask}/{args.approach}/{args.shortcut_model_name}/vectors/{args.sentence_embedder}_id2vec.tsv"
    with open(vector_file_path, "r") as fr:
        for line in fr:
            id_, vec = line.strip().split('\t')
            id2vec_disambiguated_data[id_] = vec
    return id2vec_disambiguated_data
  
def _get_gold_data_vectors():
    """
    Retrieves gold data sentence embeddings from a file and returns them as a dictionary.

    Args:
        None

    Returns:
        dict: A dictionary mapping instance IDs to their corresponding sentence embeddings.
    """
    id2vec_gold = {}

    with open(f"../data/evaluation/vectors/{args.sentence_embedder}_id2vec.tsv") as fr:
        for line in fr:
            id, definition, vec = line.strip().split('\t')
            id2vec_gold[id+definition] = vec
    return id2vec_gold


def _write_definition_ranks(definition_ranks_path, definition_ranks_list):
    if args.gpt_as_judge == False: # both in DS and DG scenarios
        with open(definition_ranks_path, mode="w") as json_file:
            json_file.write(json.dumps(definition_ranks_list, indent=4))
    else: # we add '****' to the definition choosen by the gpt_as_judge
        assert os.path.isfile(definition_ranks_path)
        gpt_candidates_list = [elem["candidates"][0] for elem in definition_ranks_list] # these are the senses choosen by gpt_as_judge
        with open(definition_ranks_path, mode="r") as json_file:
            cos_sim_data = json.load(json_file)
        for idx,item in enumerate(cos_sim_data):
            for i,elem in enumerate(item["candidates"]):
                if elem == gpt_candidates_list[idx]: item["candidates"][i] = "**** " + item["candidates"][i] + " ****"; break
        with open(definition_ranks_path, mode="w") as json_file:
            json_file.write(json.dumps(cos_sim_data, indent=4))

def _print_scores(true_labels, predicted_labels, number_of_evaluation_instances, correct, wrong):
    
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')

    print("-----")
    print("Total number of instances:", number_of_evaluation_instances)
    print("Number of correctly classified instances:", correct)
    print("Number of incorrectly classified instances:", wrong)
    print()
    print("Precision (average=micro):", precision)
    print("Recall (average=micro):", recall)
    print("F1 Score (average=micro):", f1)
    print("Accuracy:", correct/number_of_evaluation_instances)

###########################################################################################################

from statsmodels.stats.contingency_tables import mcnemar 
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
    gpt_gold_path = "../data/evaluation/ALLamended/ALLamended_preprocessed.json"
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


if __name__ == "__main__":

    mc_nemar_test("selection", "zero_shot")