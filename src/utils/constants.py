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

def init_training_config(config_filename='configs/train.yaml'):
    """Load constants from the given configuration YAML file."""
    config = load_yaml_config(config_filename)
    
    constants = {
        'MODEL_NAME': config['model']['model_name'],
        'LEARNING_RATE': float(config['hyperparameters']['learning_rate']),
        'NUM_EPOCHS': config['hyperparameters']['num_train_epochs'],
        'TRAIN_BATCH_SIZE': config['hyperparameters']['per_device_train_batch_size'],
        'EVAL_BATCH_SIZE': config['hyperparameters']['per_device_eval_batch_size'],
        'WEIGHT_DECAY': config['hyperparameters']['weight_decay'],
        'SAVE_LIMIT': config['hyperparameters']['save_total_limit'],
        'LOGGING_STEPS': config['hyperparameters']['logging_steps'],
        'BEST_MODEL_METRIC': config['hyperparameters']['metric_for_best_model'],
        'GREATER_IS_BETTER': config['hyperparameters']['greater_is_better'],
        'EVAL_STRATEGY': config['hyperparameters']['evaluation_strategy'],
        'LOGGING_STRATEGY': config['hyperparameters']['logging_strategy'],
        'SAVE_STRATEGY': config['hyperparameters']['save_strategy'],
        'LOAD_BEST_MODEL': config['hyperparameters']['load_best_model_at_end'],
        'ID2LABEL': config['labels']['id2label'],
        'LABEL2ID': config['labels']['label2id'],
        'DATA_PATH': config['hf']['data_path'],
        'SAVE_MODEL_PATH': config['hf']['save_model_path'],
    }
    return constants

