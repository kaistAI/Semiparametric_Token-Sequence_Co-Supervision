import inspect
import configs.datasets as datasets
from configs import training_config
def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, training_config):
                print(f"Warning: unknown parameter {k}")
def generate_dataset_config(train_config, kwargs):
    # names = tuple(DATASET_PREPROC.keys())
    
    # assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"
    
    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[train_config.dataset]
    update_config(dataset_config, **kwargs)
    
    return  dataset_config