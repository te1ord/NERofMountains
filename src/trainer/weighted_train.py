import torch
from datasets import load_dataset
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.constants import init_datagen_config
from utils.constants import init_training_config
from utils.model_utils import tokenize_and_align_labels, compute_metrics, get_data_collator, setup_logging

def compute_class_weights(dataset):
    """
    Compute class weights based on the class frequencies in the dataset.
    This will calculate weights for each class: 'B-Mountain', 'I-Mountain', and 'O'.
    """
    label_counts = [0, 0, 0]  # Assuming labels: 0: 'O', 1: 'B-Mountain', 2: 'I-Mountain'

    for example in dataset['train']:  # Use the training set to calculate class frequencies
        labels = example['labels']
        for label in labels:
            if label != -100:  # Skip padded tokens
                label_counts[label] += 1

    total_labels = sum(label_counts)
    class_weights = [total_labels / count for count in label_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights

def main():
    # logging
    setup_logging(log_file='../logs/training.log')

    # training config
    constants = init_training_config()

    # pre-trained model and tokenizer
    model_name = constants['MODEL_NAME']
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,  # B-Mountain, I-Mountain, O
        id2label=constants['ID2LABEL'],   
        label2id=constants['LABEL2ID'],   
        ignore_mismatched_sizes=True  
    )

    # load & tokenize dataset
    dataset = load_dataset(constants['DATA_PATH'])
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    # Compute class weights based on training set distribution
    class_weights = compute_class_weights(dataset)

    # Modify the model's loss function to use class weights
    def compute_loss_with_class_weights(outputs, labels):
        """
        Custom loss function that applies class weights during training.
        """
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.device), ignore_index=-100)
        return loss_fct(outputs.view(-1, model.config.num_labels), labels.view(-1))

    # Override the Trainer class to include the class-weighted loss function
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = compute_loss_with_class_weights(logits, labels)
            return (loss, outputs) if return_outputs else loss

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
        report_to="none"                    
    )

    # trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],
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


if __name__ == "__main__":
    main()
