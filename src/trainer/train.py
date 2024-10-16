import torch
from datasets import load_dataset
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.constants import init_datagen_config

from utils.constants import init_training_config  
from utils.model_utils import tokenize_and_align_labels, compute_metrics, get_data_collator  

# training config
constants = init_training_config()

# pre-trained 
model_name = constants['MODEL_NAME']
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(
    model_name,
    num_labels=3, # B-Mountain, I-Mountain, O
    id2label=constants['ID2LABEL'],   
    label2id=constants['LABEL2ID'],   
    ignore_mismatched_sizes=True  
)

# load & tokenize dataset
dataset = load_dataset(constants['DATA_PATH'])
tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

# data collator 
data_collator = get_data_collator(tokenizer)

# training arguments
training_args = TrainingArguments(
    output_dir='../results',
    evaluation_strategy=constants['EVAL_STRATEGY'],  
    logging_strategy=constants['LOGGING_STRATEGY'],  
    logging_steps=constants['LOGGING_STEPS'],        
    save_strategy=constants['SAVE_STRATEGY'],       
    load_best_model_at_end=constants['LOAD_BEST_MODEL'],  
    metric_for_best_model=constants['BEST_MODEL_METRIC'], 
    greater_is_better=constants['GREATER_IS_BETTER'],  
    learning_rate=constants['LEARNING_RATE'],         
    per_device_train_batch_size=constants['TRAIN_BATCH_SIZE'],  
    per_device_eval_batch_size=constants['EVAL_BATCH_SIZE'],  
    num_train_epochs=constants['NUM_EPOCHS'],         
    weight_decay=constants['WEIGHT_DECAY'],           
    save_total_limit=constants['SAVE_LIMIT'],        
    logging_dir='../logs',                
    report_to="none"                    
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,  
    compute_metrics=compute_metrics 
)

# fine-tune 
trainer.train()

# eval 
trainer.evaluate()

# save to hub
model.push_to_hub(constants['SAVE_MODEL_PATH'])
tokenizer.push_to_hub(constants['SAVE_MODEL_PATH'])
