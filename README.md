## Do Large Language Models Understand Word Meanings?

### COMMANDS 🕹️
> (go inside the *src/* folder to run)

#### Disambiguate
```bash
python3 disambiguate.py --shortcut_model_name {model} --subtask {selection,generation} --approach {zero_shot,one_shot,few_shot}
```
> If the model is **finetuned** simply add:
```bash
--is_finetuned True --checkpoint_path {path_to_checkpoint or hf_hub_model_name}
```
> and if the finetuned model has been trained on **LLMB** data, add the flag:
```bash
--LLMB True
```

#### Score
```bash
python3 score.py --shortcut_model_name {model} --subtask {selection,generation} --approach {zero_shot,one_shot,few_shot} --pos {NOUN,ADJ,VERB,ADV,ALL}
```
> For evaluating in **generation** setting you need to add: 
```bash
--sentence_embedder all-mpnet-base-v2
```
> in the case of cosine similarity and 
```bash
--llm_as_judge True
```
> when we want to use LLM *as a judge*.

> If the model is **finetuned** simply add:
```bash
--is_finetuned True
```
> and if the finetuned model has been trained on **LLMB** data, add the flag:
```bash
--LLMB True
```

#### Finetune
```bash
python3 finetune.py --shortcut_model_name {model} --subtask {selection,generation} --epochs 8 --batch_size 8
```
> If we want to train on **SemCor + LLMB**, add the flag:
```bash
--LLMB True
```
----------------------------------------------------------------

### SotA WSD systems
> To run the evaluation and see the results of state-of-the-art WSD systems such as **ConSeC** and **ESC** we built two *Docker* images.

#### Requirements
To successfully run both systems with GPU acceleration, follow these steps:
*  **Install Docker**: first, ensure that Docker is installed on your system. Docker provides the containerization platform required to run both environments independently and reproducibly.
*  **Install NVIDIA Container Toolkit**: to enable GPU support inside Docker containers, you need to install the *nvidia-container-toolkit*. This toolkit allows Docker to interface with your NVIDIA drivers and provides the necessary runtime components to access GPUs from within containers.
*  **Choose the Appropriate NVIDIA CUDA Image**: select a CUDA-enabled base image that matches your system’s driver and software requirements. The image must be compatible with both the CUDA version you intend to use and the Ubuntu/Debian version you’re comfortable working with. For example, in my setup, I opted for: ```nvidia/cuda:11.0.3-base-ubuntu20.04```. This image suits my environment because my system already supports CUDA 11.0 and is built on Ubuntu 20.04. You should choose an image that aligns with your hardware capabilities and target software stack.

> If the provided images are compatible with your environment, you can simply pull them from Docker Hub using the following commands:
```bash
# pull custom images from docker hub
docker pull lavalloone/wsd_sota:consec
docker pull lavalloone/wsd_sota:esc
```
> However, if you need to use a different base image (e.g., to match your system’s CUDA or OS version), you’ll need to modify the ```FROM``` line in the ```dockerfile_consec``` and ```dockerfile_esc``` files located in the ```wsd_sota``` directory. After updating the base image, build the Docker images locally using:
```bash
# build docker images from custom dockerfiles
docker build -f dockerfile_consec -t consec .
docker build -f dockerfile_esc -t esc .
```

#### Run Evaluation
To start the evaluation process using the `consec` and `esc` Docker images, first run the containers with GPU access:
```bash
# run the consec image
docker run --gpus all -it --rm --entrypoint /bin/bash consec
# run the esc image
docker run --gpus all -it --rm --entrypoint /bin/bash esc
```
Once inside the ```consec``` container, run the evaluation pipeline with:
```bash
PYTHONPATH=$(pwd) python src/scripts/model/raganato_evaluate.py \
model.model_checkpoint=experiments/released-ckpts/consec_semcor_normal_best.ckpt \
test_raganato_path=data/WSD_Evaluation_Framework/Evaluation_Datasets/ALLamended/ALLamended
```
For the ```esc``` container, use the following command instead:
```bash
PYTHONPATH=$(pwd) python esc/predict.py \
--ckpt experiments/escher_semcor_best.ckpt \
--dataset-paths data/WSD_Evaluation_Framework/Evaluation_Datasets/ALLamended/ALLamended.data.xml \
--prediction-types probabilistic \
--evaluate
```
