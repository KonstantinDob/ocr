import cv2
import yaml
import numpy as np
from typing import Dict, List, Any

from ocr_recognition.inference import InferenceOCRRec
from ocr_detection.inference import InferenceOCRDet
from ocr.builder.openvino_inference import (
    InferenceRecOV,
    InferenceDetOV
)

from ocr.modules.common import BadImageException
from ocr.builder.onnx_inference import (
    InferenceRecONNX,
    InferenceDetONNX
)


class InferenceOCR:
    """Inference OCR pipeline.

    In current pipeline is used the modules:
    ocr_detection: https://github.com/KonstantinDob/ocr_detection,
    ocr_recognotion: https://github.com/KonstantinDob/ocr_recognition.
    Before run, you should train/load weights and prepare configs in
    configs folder.
    If you want to speed up model with ONNX, OpenVINO need to convert
    PyTorch-like weights with ocr/modules.

    Raises:
        BadImageException: Raised when incorrect image type was loaded
            to inference. Should be RGB np.uint8.
    """

    def __init__(self):
        self.rec_model: Any = None
        self.det_model: Any = None

        self.rec_config: Dict[str, Any] = dict()
        self.det_config: Dict[str, Any] = dict()
        self.config: Dict[str, Any] = dict()

        self.device: str = ''

        self._create_modules()

    def _create_modules(self):
        """Load configs and models."""
        self._load_configs()

        if self.config['speed_up'] in ['cpu', 'cuda']:
            self.rec_model = InferenceOCRRec(self.rec_config)
            self.det_model = InferenceOCRDet(self.det_config)
        elif self.config['speed_up'] == 'onnx':
            self.rec_model = InferenceRecONNX(self.rec_config)
            self.det_model = InferenceDetONNX(self.det_config)
        elif self.config['speed_up'] == 'openvino':
            self.rec_model = InferenceRecOV(self.rec_config)
            self.det_model = InferenceDetOV(self.det_config)
        else:
            raise KeyError('Incorrect speed up type!')

    def _load_configs(self):
        """Load configs from configs."""
        with open('configs/inference.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
            self.config['speed_up'] = self.config['speed_up'].lower()
            self.device = 'cuda' if self.config['speed_up'] == 'cuda' \
                else 'cpu'
        with open('configs/recognition_inference.yaml', 'r') as file:
            self.rec_config = yaml.safe_load(file)
            self.rec_config['device'] = self.device
        with open('configs/detection_inference.yaml', 'r') as file:
            self.det_config = yaml.safe_load(file)
            self.det_config['device'] = self.device

    def _check_image(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise BadImageException("Incorrect image type!"
                                    "Expected np.ndarray, "
                                    f"got {type(image)}")
        if len(image.shape) != 3:
            raise BadImageException("Image should be RGB!")
        if image.shape[2] != 3:
            raise BadImageException("Image should be RGB!")

    def _create_crops(self, prediction: List[np.ndarray],
                      image: np.ndarray) -> Dict[int, np.array]:
        """Make crops based on detection model.

        Too small crops are not used further.
        """
        crops = dict()
        for idx, pred in enumerate(np.array(prediction)):
            x_1 = np.min(pred[:, 1])
            x_2 = np.max(pred[:, 1])
            y_1 = np.min(pred[:, 0])
            y_2 = np.max(pred[:, 0])

            if abs(x_2 - x_1) < self.config['min_crop_size'][0] or \
                    abs(y_2 - y_1) < self.config['min_crop_size'][1]:
                continue

            crop = image[x_1:x_2, y_1:y_2, :]
            crops[idx] = crop
        return crops

    def _visualize_prediction(self, image: np.ndarray,
                              det_prediction: List[np.ndarray],
                              text: Dict[int, str]) -> np.ndarray:
        """Show detected words on the input image."""
        # Show text.
        for key, val in text.items():
            prediction = det_prediction[key]
            x = np.min(prediction[:, 0])
            y = np.min(prediction[:, 1])
            cv2.putText(image, val, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Remove odd detection object.
        idx_set: set = set(range(len(det_prediction)))
        same_idx = list(idx_set.intersection(text.keys()))
        det_prediction = [det_prediction[idx] for idx in same_idx]
        image = self.det_model.visualize(image=image,
                                         prediction=np.copy(
                                             det_prediction))
        return image

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Make prediction on the input image.

        Input image should be RGB numpy array.
        """
        self._check_image(image)

        det_prediction = self.det_model.predict(image)
        crops = self._create_crops(det_prediction, image)
        text = {key: self.rec_model.predict(val) for key, val
                in crops.items()}

        output = self._visualize_prediction(image, det_prediction, text)
        return output
