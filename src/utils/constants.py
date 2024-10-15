import yaml

def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)  

def init_datagen_config(config_filename='configs/datagen.yaml'):
    """Load constants from the given configuration YAML file."""

    gen_config = load_yaml_config(config_filename)['datagen_config']
    paths_config = load_yaml_config(config_filename)['paths']

    constants = {
        'MOUNTAINS_NAMES_PATH': paths_config['mountains_path'],
        'SAVE_DATASET_PATH': paths_config['save_dataset_path'],
        'SAVE_PROCESSED_PATH': paths_config['save_processed_path'],
        'GENERATOR_MODEL': gen_config['generator_model'],
        'MIN_SAMPLES': gen_config['min_samples'],
        'MAX_SAMPLES': gen_config['max_samples'],
        'TEMPERATURE': gen_config['temperature'],
        'WNUT_DATA_PATH': paths_config['wnut_data_path'],
        'PROCESSED_WNUT_PATH': paths_config['processed_wnut_path'],
        'FEW-NERD_BALANCED_TRAIN_PATH' : paths_config['few_nerd_balanced_train'],
        'FEW-NERD_BALANCED_VAL_PATH' : paths_config['few_nerd_balanced_val'],
        'FEW-NERD_BALANCED_TEST_PATH' : paths_config['few_nerd_balanced_test'],
        'FINAL_DATASET_PATH': paths_config['final_dataset_path']
    }
    return constants

