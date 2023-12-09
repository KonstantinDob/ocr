import yaml
from os.path import join
from typing import Dict, Any


def load_config(model_type: str) -> Dict[str, Any]:
    """Load inference config.

    Args:
        model_type (str): What model configs should be loaded. Related to folder name in
            "./data" folder.

    Returns:
        dict of str: Any: Config with model/data parameters.
    """
    with open(join("./configs", f"{model_type}_inference.yaml"), "r") as file:
        data_config = yaml.safe_load(file)
    return data_config
