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
    gold_data_path = "../data/evaluation/ALLamended_preprocessed.json"
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

def _print_log(subtask, approach, shortcut_model_name, last_prompt, n_instances_processed):
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

def _generate_gold_data_vectors(subtask, sentence_embedder):
    """
    Generates sentence embeddings for gold data and saves them to a file.

    Args:
        subtask (str)

    Returns:
        None
    """
    gold_vector_file_path = f"../data/evaluation/vectors/{sentence_embedder}_id2vec.tsv"
    if os.path.exists(gold_vector_file_path):
        print("Gold vectors already exist")
        return None
    gold_vector_folder_path = f"../data/evaluation/vectors"
    if not os.path.exists(gold_vector_folder_path):
        os.makedirs(gold_vector_folder_path)

    print("Generating vectors from gold data")
    embedder = SentenceTransformer(f'sentence-transformers/{sentence_embedder}')
    data = _get_gold_data(subtask)

    gold_vector_file_path = f"../data/evaluation/vectors/{sentence_embedder}_id2vec.tsv"
    with open(gold_vector_file_path, "w") as fw:
        for el in tqdm(data, total=len(data)):
            id_ = el["id"]
            definitions = el["definitions"]
            for definition in definitions:
                vec = embedder.encode(definition).tolist()
                vec = " ".join([str(x) for x in vec])
                fw.write(f"{id_}\t{definition}\t{vec}\n")

def _generate_disambiguated_data_vectors(subtask, approach, shortcut_model_name, sentence_embedder, is_finetuned, disambiguated_data_path, len_gold):
    """
    Generates sentence embeddings for disambiguated data and saves them to a file.

    Args:
        disambiguated_data_path (str): the path to the data disambiguated file 
        len_gold (int): number of instances in the gold file
        is_finetuned (bool): if the model is finetuned or not
    Returns:
        None
    """
    vector_file_path = f"../data/{subtask}/{approach}/finetuned_{shortcut_model_name}/vectors/{sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{subtask}/{approach}/{shortcut_model_name}/vectors/{sentence_embedder}_id2vec.tsv"
    if os.path.exists(vector_file_path):
        with open(vector_file_path, "r") as fr:
            if len(fr.readlines()) != len_gold:
                print("Missing vectors")
                exit()
            print("Disambiguated data vectors already exist")
            return None
    vector_folder_path = f"../data/{subtask}/{approach}/finetuned_{shortcut_model_name}/vectors" if is_finetuned else f"../data/{subtask}/{approach}/{shortcut_model_name}/vectors"
    if not os.path.exists(vector_folder_path):
        os.makedirs(vector_folder_path)          

    print("Generating vectors from:", disambiguated_data_path)
    embedder = SentenceTransformer(f'sentence-transformers/{sentence_embedder}')

    with open(disambiguated_data_path) as fr:
        data = json.load(fr)

    vector_file_path = f"../data/{subtask}/{approach}/finetuned_{shortcut_model_name}/vectors/{sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{subtask}/{approach}/{shortcut_model_name}/vectors/{sentence_embedder}_id2vec.tsv"
    id2vec_dd = {}
    with open(vector_file_path, "w") as fw:
        for el in tqdm(data, total=len(data)):
            id_ = el["instance_id"]
            answer = el["answer"]
            vec = embedder.encode(answer)
            id2vec_dd[id_] = " ".join([str(x) for x in vec.tolist()])
            vec = " ".join([str(x) for x in vec])
            fw.write(f"{id_}\t{vec}\n")

def _get_disambiguated_data_vectors(subtask, approach, shortcut_model_name, sentence_embedder, is_finetuned):
    """
    Retrieves the disambiguated data sentence embeddings from a file and returns them as a dictionary.

    Args:
        is_finetuned (bool): if the model is finetuned or not

    Returns:
        dict: A dictionary mapping instance IDs to their corresponding sentence embeddings.
    """
    id2vec_disambiguated_data = {}

    vector_file_path = f"../data/{subtask}/{approach}/finetuned_{shortcut_model_name}/vectors/{sentence_embedder}_id2vec.tsv" if is_finetuned else f"../data/{subtask}/{approach}/{shortcut_model_name}/vectors/{sentence_embedder}_id2vec.tsv"
    with open(vector_file_path, "r") as fr:
        for line in fr:
            id_, vec = line.strip().split('\t')
            id2vec_disambiguated_data[id_] = vec
    return id2vec_disambiguated_data
  
def _get_gold_data_vectors(sentence_embedder):
    """
    Retrieves gold data sentence embeddings from a file and returns them as a dictionary.

    Args:
        None

    Returns:
        dict: A dictionary mapping instance IDs to their corresponding sentence embeddings.
    """
    id2vec_gold = {}

    with open(f"../data/evaluation/vectors/{sentence_embedder}_id2vec.tsv") as fr:
        for line in fr:
            id, definition, vec = line.strip().split('\t')
            id2vec_gold[id+definition] = vec
    return id2vec_gold


def _write_definition_ranks(definition_ranks_path, definition_ranks_list, gpt_as_judge):
    if gpt_as_judge == False: # both in DS and DG scenarios
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