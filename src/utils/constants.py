import yaml

def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)  

def init_datagen_config(config_filename='configs/datagen.yaml'):
    """Load constants from the given configuration YAML file."""

    config = load_yaml_config(config_filename)['datagen_config']

    constants = {
        'MOUNTAINS_NAMES_PATH': config['mountains_path'],
        'SAVE_DATASET_PATH': config['save_dataset_path'],
        'GENERATOR_MODEL': config['generator_model'],
        'MIN_SAMPLES': config['min_samples'],
        'MAX_SAMPLES': config['max_samples'],
        'TEMPERATURE': config['temperature'],

    }
    return constants

