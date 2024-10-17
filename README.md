# Mountain NER (Named Entity Recognition) Project

This project aims to build a Named Entity Recognition (NER) model for detecting mentions of mountains in texts. It uses a fine-tuned BERT model to classify tokens into three categories: `B-Mountain` (beginning of a mountain entity), `I-Mountain` (inside a mountain entity), and `O` (other).

The project includes functionality to train the NER model on a custom dataset, generate synthetic data for additional training, and perform inference on new input sentences.

---

## Project Structure

```bash
NERofMountains/
│
├── dataset/                     # datasets used for training and evaluation
│   ├── few-nerd/                
│   ├── resulting_dataset/       
│   ├── synthetic/               
│   └── wnut16/                 
│
├── logs/                        # logs generated during training 
│   └── training.log             
│
├── src/                         
│   ├── configs/                 # configs for dataset creation and model training/inference
│   │   ├── datagen.yaml         
│   │   └── model.yaml           
│   ├── notebooks/               
│   │   ├── dataset.ipynb        # dataset creation 
│   │   ├── NERofMountainsDemo.ipynb # demo
│   │   └── NERofMountainsTrainer.ipynb # kaggle notebook for trainining
│   ├── SyntheticDataGenerator/  #  synthetic data generation
│   │   └── llm_generator.py     
│   ├── trainer/                
│   │   ├── inference.py         # inference ()
│   │   └── train.py             
│   └── utils/                   # utils functions and constants used in training and inference
│       ├── constants.py         # constants used throughout the project
│       └── model_utils.py       # helper functions for model loading and preprocessing
│
├── README.md                    
└── requirements.txt             
```

---

## Solution Overview

The project builds an NER model to detect mountain names in text. It uses [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) as the base model and fine-tunes it for token classification tasks. The project includes:

1. **Dataset Preparation**: Final dataset was created from 3 different sources - [few-nert dataset](https://huggingface.co/datasets/DFKI-SLT/few-nerd), [wnut16](https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16) and synthetic data generated in llm_generator.py file. From few-nerd I selected samples with mountains as well as without them. Synthetic dataset contains only samples with montains. WNUT 16 has no sentences containing mountains and is used as augmentation. The resulting dataset can be found [here](https://huggingface.co/datasets/telord/mountains-ner-dataset). 

![alt text](token_imbalance.jpg)

The resulting dataset it almost 50/50 balanced in terms of samples with/without mountains, but target tokens has very imbalanced distribution
O: 94.45%
B-Mountain: 2.82%
I-Mountain: 2.73%

   
2. **Model Training**: 
I chose the bert base model fine-tuned on [CoNLL-2003 Named Entity Recognition](https://aclanthology.org/W03-0419.pdf) and fine-tuned it on created dataset. My experiments included training it with and without class weights and I got almost the same score of 0.91 on validation set for both ways.

#### Training Configuration

The training settings are defined in `src/configs/model.yaml`. You can modify hyperparameters like `num_train_epochs`, `learning_rate`, and others to suit your needs.

Example configuration (`model.yaml`):

```yaml
model:
  model_name: "dslim/bert-base-NER"

hyperparameters:
  learning_rate: 2e-5
  num_train_epochs: 10
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  weight_decay: 0.01
  save_total_limit: 3
  logging_steps: 100
  metric_for_best_model: "f1"
  greater_is_better: true
  evaluation_strategy: "epoch"
  logging_strategy: "steps"
  save_strategy: "epoch"
  load_best_model_at_end: true

hf:
  data_path: "telord/ner-mountains-dataset"
  save_model_path: "telord/ner-mountains-model"
```

The final model's weights can be found [here]().


3. **Inference**: Once the model is trained, it can be used to predict mountain entities in new text inputs. The `inference.py` script allows for command-line predictions or integration into Jupyter notebooks. It is designed for single example inference but can be easily adapted for large-scale inference if needed.


---

## Setup

### 1. Clone the repository:

```bash
git clone <your-repo-url>
cd NEROFMOUNTAINS
```

### 2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Setup Configuration:

You can customize the training and data generation settings via the configuration files located in `src/configs/`.

- **`model.yaml`**: Adjusts hyperparameters for training the NER model (epochs, batch size, learning rate, etc.).
- **`datagen.yaml`**: Defines parameters for generating synthetic data.

### 4. src folder:
The code is designed to be easily run from source folder in the following way:

```bash
python folder-where-your-script-is-located/script.py
```


### 5. Training the Model:

To train the model, run:

```bash
python trainer/train.py
```

This will:
- Load the dataset
- Fine-tune the model
- Save the best model and tokenizer to the `./results` directory and push it to hub defined in `model.yaml`

### 6. Running Inference:

To run inference on a specific sentence using the trained model, use:

```bash
python trainer/inference.py --sentence "Denali is the tallest mountain in North America."
```

This will output the detected mountain entities in the sentence:

---
Denali: B-Mountain
is: O
the: O
tallest: O
mountain: O
in: O
North: O
America: O
.: O
---

## Results and Performance

The model achieves good accuracy in recognizing mountain names across different contexts. For evaluation, precision, recall, and F1 score metrics are logged after each epoch.

Example metrics from training:
- **Accuracy**: 0.981
- **Precision**: 0.857
- **Recall**: 0.871
- **F1 Score**: 0.864

---

## Inference Demo

To demonstrate inference in a Jupyter notebook, use the `NERofMountainsDemo.ipynb` notebook in the `notebooks/` directory. This shows how to pass sentences through the trained model and display entity predictions on samples from different parts of my dataset as well as on ones, which were not present during training and were designed to test model in complicated scenarios.