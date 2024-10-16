import torch
from transformers import BertForTokenClassification, BertTokenizerFast
import os
import sys
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.constants import init_training_config  # Import your config initialization function

def load_model_for_inference(config):
    """
    Load the fine-tuned model and tokenizer for NER inference from the provided config.

    Args:
    config (dict): Configuration dictionary with model and tokenizer paths.

    Returns:
    model (torch.nn.Module): The fine-tuned NER model.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to preprocess the input text.
    """
    model_name = config['SAVE_MODEL_PATH']
    model = BertForTokenClassification.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    
    return model, tokenizer

def post_process_tokens_and_labels(tokens, predicted_labels):
    """
    Post-process tokens and labels by merging subwords and removing special tokens like [CLS] and [SEP].

    Args:
    tokens (list): List of tokenized input words.
    predicted_labels (list): List of predicted labels.

    Returns:
    final_tokens (list): List of final tokens with subwords merged.
    final_labels (list): List of final labels corresponding to each token.
    """
    final_tokens, final_labels = [], []

    for token, label in zip(tokens, predicted_labels):
        if token not in ['[CLS]', '[SEP]']:  # Skip special tokens
            if token.startswith("##"):
                final_tokens[-1] += token[2:]  # Merge subwords
            else:
                final_tokens.append(token)
                final_labels.append(label)

    return final_tokens, final_labels


def predict_ner_labels(sentence, model, tokenizer, device):
    """
    Predicts NER labels for a given sentence using the fine-tuned model and tokenizer.
    
    Args:
    sentence (str): The input sentence.
    model (torch.nn.Module): The fine-tuned NER model.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to tokenize the input.
    device (torch.device): The device (CPU or GPU) on which to run the model.
    """
    
    # tokenize 
    inputs = tokenizer(sentence.split(), return_tensors="pt", is_split_into_words=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs).logits

    # predicted labels
    predictions = outputs.argmax(dim=-1).squeeze().tolist()
    predicted_labels = [model.config.id2label[pred] for pred in predictions]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    # post-process
    final_tokens, final_labels = post_process_tokens_and_labels(tokens, predicted_labels)

    for word, label in zip(final_tokens, final_labels):
        print(f"{word}: {label}")

if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser(description="NER Prediction using fine-tuned BERT model.")
    parser.add_argument("--sentence", type=str, required=True, help="Input sentence for NER prediction.")
    args = parser.parse_args()

    # config
    config = init_training_config()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model & tokenizer 
    model, tokenizer = load_model_for_inference(config)
    model.to(device)
    
    # run 
    # python trainer/inference.py  --sentence "your sentence"
    predict_ner_labels(args.sentence, model, tokenizer, device)    
