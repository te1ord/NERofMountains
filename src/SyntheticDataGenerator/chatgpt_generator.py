import os
import json
from openai import OpenAI
import re  # Import regex module

# Setting up the Nvidia API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key=os.environ["NVIDIA_API_KEY"]
)
MODEL = "meta/llama-3.1-405b-instruct"

# Function to save responses to a file
def save_to_file(filename: str, dataset: list):
    with open(filename, 'w') as f:  # Open in write mode to overwrite existing content
        json.dump(dataset, f, indent=2)  # Save dataset as a JSON array

# Function to generate diverse sentences about mountains
def generate_diverse_sentences(client, mountain_name, num_sentences=10, temperature=0.7):
    prompt = (f"Generate {num_sentences} unique sentences about the mountain '{mountain_name}'. "
              "These sentences should vary in context such as travel, geography, personal experience, or news. "
              "Each sentence should look natural, contain the mountain name, and not be similar to the others. "
              "Make sure the sentences are well-formed and unique.")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )

    # Clean up the sentences: split by newline
    raw_sentences = response.choices[0].message.content.strip().split('\n')
    
    # Keep only relevant sentences and remove leading numbers
    cleaned_sentences = []
    for sentence in raw_sentences:
        # Exclude unwanted phrases
        if sentence and not (sentence.startswith("Here are") or sentence.startswith("I hope")):
            # Remove leading numbers and any trailing period or space
            cleaned_sentence = re.sub(r'^\s*\d+\.\s*', '', sentence)
            cleaned_sentences.append(cleaned_sentence.strip())

    return cleaned_sentences

# Function to generate an annotated dataset for a list of mountains
def create_annotated_dataset(client, mountain_list, num_samples_per_mountain=10):
    dataset = []
    for mountain in mountain_list:
        sentences = generate_diverse_sentences(client, mountain, num_sentences=num_samples_per_mountain)
        dataset.append({'mountain': mountain, 'sentences': sentences})
        print(f"Generated sentences for {mountain}")
    return dataset

# Function to load and correctly parse the dataset from the saved file
def load_dataset(filename: str):
    with open(filename, 'r') as f:
        dataset = json.load(f)  # Load dataset as JSON directly
    return dataset

# Example list of mountains
mountains = ['Everest', 'Kilimanjaro', 'Mont Blanc', 'Denali', 'Aconcagua', 'Elbrus']

# Check if the dataset file already exists
dataset_file = '../dataset/mountain_ner_dataset.txt'

if not os.path.exists(dataset_file):
    # Generate the annotated dataset
    annotated_dataset = create_annotated_dataset(client, mountains, num_samples_per_mountain=10)
    # Save the generated dataset to a file
    save_to_file(dataset_file, annotated_dataset)
else:
    print(f"{dataset_file} already exists. Skipping generation.")

# Example usage of the load dataset function
loaded_data = load_dataset(dataset_file)
print(loaded_data)

# import os
# import json
# from openai import OpenAI

# # Setting up the Nvidia API client
# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1", 
#     api_key=os.environ["NVIDIA_API_KEY"]
# )
# MODEL = "meta/llama-3.1-405b-instruct"

# # Function to save responses to a file
# def save_to_file(filename: str, dataset: list):
#     with open(filename, 'w') as f:  # Open in write mode to overwrite existing content
#         json.dump(dataset, f, indent=2)  # Save dataset as a JSON array

# # Function to generate diverse sentences about mountains
# def generate_diverse_sentences(client, mountain_name, num_sentences=10, temperature=0.7):
#     prompt = (f"Generate {num_sentences} unique sentences about the mountain '{mountain_name}'. "
#               "These sentences should vary in context such as travel, geography, personal experience, or news. "
#               "Each sentence should look natural, contain the mountain name, and not be similar to the others. "
#               "Here are some examples of different contexts: "
#               f"'The team climbed {mountain_name} last year during their expedition.' "
#               f"'{mountain_name} is one of the most dangerous peaks in the world.' "
#               "Make sure the sentences are well-formed and unique.")

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=temperature,
#     )
    
#     return response.choices[0].message.content.strip().split('\n')

# # Function to generate an annotated dataset for a list of mountains
# def create_annotated_dataset(client, mountain_list, num_samples_per_mountain=10):
#     dataset = []
#     for mountain in mountain_list:
#         sentences = generate_diverse_sentences(client, mountain, num_sentences=num_samples_per_mountain)
#         # Add sentences without additional numbering or an introductory line
#         dataset.append({'mountain': mountain, 'sentences': sentences})
#         print(f"Generated sentences for {mountain}")
#     return dataset

# # Function to load and correctly parse the dataset from the saved file
# def load_dataset(filename: str):
#     with open(filename, 'r') as f:
#         dataset = json.load(f)  # Load dataset as JSON directly
#     return dataset

# # Example list of mountains
# mountains = ['Everest', 'Kilimanjaro', 'Mont Blanc', 'Denali', 'Aconcagua', 'Elbrus']

# # Check if the dataset file already exists
# dataset_file = 'mountain_ner_dataset.txt'
# if not os.path.exists(dataset_file):
#     # Generate the annotated dataset
#     annotated_dataset = create_annotated_dataset(client, mountains, num_samples_per_mountain=10)
#     # Save the generated dataset to a file
#     save_to_file(dataset_file, annotated_dataset)
# else:
#     print(f"{dataset_file} already exists. Skipping generation.")

# # Example usage of the load dataset function
# loaded_data = load_dataset(dataset_file)
# print(loaded_data)

