import pandas as pd
import json
from openai import OpenAI
import re
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.constants import init_datagen_config

# setting up the Nvidia API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key=os.environ["NVIDIA_API_KEY"] # use your api key (export NVIDIA_API_KEY='your_api_key_here') access at https://build.nvidia.com/nim
)

# extract mountain names 
def extract_mountain_names(csv_file: str):
    df = pd.read_csv(csv_file)
    mountain_names = df['Mountain'].tolist()  
    return mountain_names

# save responses
# def save_to_file(filename: str, dataset: list):
#     with open(filename, 'w') as f:
#         json.dump(dataset, f, indent=2)  

# save each iteration to prevent data loss in case of running out of free credits 
def save_to_file(filename: str, dataset: list):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
        existing_data.extend(dataset)
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
    else:
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)        

# generate sentences about mountains
def generate_diverse_sentences(client, mountain_name, model, num_sentences, temperature):
    prompt = (f"Generate {num_sentences} unique sentences about the mountain '{mountain_name}'. "
              "These sentences should vary in context such as travel, geography, personal experience, or news. "
              "Each sentence should look natural, contain the mountain name, and not be similar to the others. "
              "Make sure the sentences are well-formed and unique."
              "Your output must contain only the sentences without any additional information meaning you must not add even comments")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,    
    )
    
    # remove unnecessary content from llm output
    raw_sentences = response.choices[0].message.content.strip().split('\n')
    cleaned_sentences = []
    for sentence in raw_sentences:
        if sentence == "":
            continue
    #     if mountain_name in sentence:
    #         cleaned_sentences.append(sentence)
        #if sentence and not (sentence.startswith("Here") or sentence.startswith("Let me know") or sentence.startswith("I hope")):
        cleaned_sentence = re.sub(r'^\s*\d+\.\s*', '', sentence)
        cleaned_sentences.append(cleaned_sentence.strip())

    return cleaned_sentences #raw_sentences 

# generate an annotated dataset for a list of mountains
def create_annotated_dataset(client, mountain_list, model, save_path, min_samples=0, max_samples=3, temperature=0.7):
    dataset = []
    for mountain in mountain_list:
        # randomize the number of sentences per mountain
        num_samples_per_mountain = random.randint(min_samples, max_samples)
        if num_samples_per_mountain == 0:
            continue
        sentences = generate_diverse_sentences(client, mountain, model, num_sentences=num_samples_per_mountain, temperature=temperature)
        mountain_data = {'mountain': mountain, 'sentences': sentences}
        dataset.append(mountain_data)
        # print(f"Generated {num_samples_per_mountain} sentences for {mountain}")

        if save_path:
            save_to_file(save_path, [mountain_data])  # Save each mountain's data separately
            print(f"Generated and saved {num_samples_per_mountain} sentences for {mountain}")
    return dataset


if __name__ == "__main__":

    CONFIG = init_datagen_config()

    mountain_names = extract_mountain_names(CONFIG['MOUNTAINS_NAMES_PATH'])
    print(len(mountain_names))

    if not os.path.exists(CONFIG['SAVE_DATASET_PATH']):
        annotated_dataset = create_annotated_dataset(client, mountain_names, model=CONFIG['GENERATOR_MODEL'], save_path=CONFIG['SAVE_DATASET_PATH'],
                                                      min_samples=CONFIG['MIN_SAMPLES'], max_samples=CONFIG['MAX_SAMPLES'], temperature=CONFIG['TEMPERATURE'])
        # save_to_file(CONFIG['SAVE_DATASET_PATH'], annotated_dataset)
    else:
        print(f"{CONFIG['SAVE_DATASET_PATH']} already exists. Skipping generation.")

