## Do Large Language Models Understand Word Meanings? 🧪 An Experimental Study

### COMMANDS 🕹️
> (go inside the *src/* folder to run)

#### Disambiguate
```
python3 disambiguate.py --shortcut_model_name {model} --subtask {selection,generation} --approach {zero_shot,one_shot,few_shot}
```
> If the model is **finetuned** simply add
```
--is_finetuned True --checkpoint_path {path_to_checkpoint or hf_hub_model_name}
```

#### Score
```
python3 score.py --shortcut_model_name {model} --subtask {selection,generation} --approach {zero_shot,one_shot,few_shot} --pos {NOUN,ADJ,VERB,ADV,ALL}
```
> For evaluating in **generation** setting you need to add: 
```
--sentence_embedder all-mpnet-base-v2
```
> in the case of cosine similarity and 
```
--llm_as_judge True
```
> when we want to use LLM *as a judge*.

> If the model is **finetuned** simply add
```
--is_finetuned True
```

#### Finetune
```
python3 finetune.py --shortcut_model_name {model} --subtask {selection,generation} --epochs 8 --batch_size 8
```