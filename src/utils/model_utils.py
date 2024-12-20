from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging
import os

import logging
import sys
import os

# logging 
def setup_logging(log_file="logs/training.log", log_level=logging.INFO):
    """
    Setup basic logging to output logs to both the console and a log file.

    Args:
    log_file (str): The name of the log file.
    log_level (logging.level): The logging level to be used (default: logging.INFO).
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # file
    os.makedirs(os.path.dirname(log_file), exist_ok=True) 
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logging setup complete.")

# imput prepcoessing
def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes the input text and aligns the labels for token classification tasks.
    
    Args:
    examples (dict): Dictionary with 'tokens' and 'labels'.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.

    Returns:
    dict: Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        is_split_into_words=True,
        padding='max_length',   # padding to the max sequence length
        truncation=True,        # truncate if exceed max length
        max_length=512,         # maximum length for input 
        return_tensors="pt"
    )
    
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  
            else:
                label_ids.append(-100)  
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# metrics
def compute_metrics(pred):
    """
    Computes metrics (precision, recall, f1-score, accuracy) for evaluation.

    Args:
    pred (transformers.EvalPrediction): Predictions from the model evaluation.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # remove ignored index (-100)
    true_labels = [label for label in labels.flatten() if label != -100]
    true_preds = [pred for label, pred in zip(labels.flatten(), preds.flatten()) if label != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='macro')
    acc = accuracy_score(true_labels, true_preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# data collator
def get_data_collator(tokenizer):
    """
    Returns a data collator for token classification.
    
    Args:
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for token classification.

    Returns:
    transformers.DataCollatorForTokenClassification: The data collator.
    """
    return DataCollatorForTokenClassification(tokenizer=tokenizer)
