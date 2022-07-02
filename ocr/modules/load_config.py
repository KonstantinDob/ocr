import yaml
from os.path import join
from typing import Dict, Any


def load_config(model_type: str) -> Dict[str, Any]:
    """Load inference config.

    Args:
        model_type (str): What model configs should be loaded.
            Related to folder name in "./data" folder.

    Returns:
        Dict[str, Any]: Config with model/data parameters.
    """
    config_path = join('./data', model_type)
    with open(join(config_path, 'inference.yaml'), 'r') as file:
        data_config = yaml.safe_load(file)
    return data_config
