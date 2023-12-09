import torch
import pytest
import ruamel.yaml
import numpy as np
from typing import List, Any

from ocr.inference import InferenceOCR
from ocr.modules.common import BadImageException

# TODO: Implement a weight loading module for testing inference.
# Once implemented, consider removing the pytest skip from all tests.


class TestInference:

    @pytest.mark.parametrize("model_type", ["cpu", "cuda", "onnx", "openvino"])
    def test_init(self, model_type: str):
        """Test inference initialization.

        Before test, you need to prepare weights for every speed up
        type.

        Args:
            model_type (str): What model speed up type will be tested.
        """
        pytest.skip("No model weights")
        yaml = ruamel.yaml.YAML()
        with open("configs/inference.yaml", "r") as file:
            config = yaml.load(file)
            config["speed_up"] = model_type
        with open("configs/inference.yaml", "w") as file:
            yaml.dump(config, file)

        inference = InferenceOCR()

        assert inference.rec_model is not None
        assert inference.det_model is not None

        assert inference.config["speed_up"] == model_type

    @pytest.mark.parametrize(
        "data_shape, created",
        [([50, 30, 3], True), ([100, 100, 3], True),
         ([50, 30, 4], False), ([50, 30, 2], False),
         ([50, 30], False), ([50, 30, 3, 3], False)]
    )
    def test_shape(self, data_shape: List[int], created: bool):
        """Test inference for input data shape"""
        pytest.skip("No model weights")
        inference = InferenceOCR()
        data = np.zeros(data_shape, dtype=np.uint8)
        try:
            inference.predict(data)
            assert created
        except BadImageException:
            assert not created

    @pytest.mark.parametrize(
        "data, created",
        [(np.zeros([50, 50, 3], dtype=np.uint8), True),
         ([100, 100, 3], False),
         (torch.zeros([50, 100, 3]), False)]
    )
    def test_data_type(self, data: Any, created: bool):
        """Test inference for input data type"""
        pytest.skip("No model weights")
        inference = InferenceOCR()
        try:
            inference.predict(data)
            assert created
        except BadImageException:
            assert not created
