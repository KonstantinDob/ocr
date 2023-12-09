import onnxruntime
import numpy as np
from typing import List

from ocr_recognition.visualizers.visualizer import to_rgb
from ocr_detection.visualizers.visualizer import VisualDrawer

from ocr_recognition.builder.base_inference import BaseInferenceOCRRec
from ocr_detection.builder.base_inference import BaseInferenceOCRDet


class InferenceRecONNX(BaseInferenceOCRRec):
    """Inference Recognition model converted to ONNX."""

    def __init__(self, *args, **kwargs):
        """Inference OCR Recognition model constructor with ONNX.

        Args:
            *args: Positional args for parent class.
            **kwargs: Keyword args for parent class.
        """
        super().__init__(*args, **kwargs)

    def _create_modules(self) -> None:
        """Create ONNX model."""
        self._load_vocabulary()

        model_path = self.config["pretrained"][:-3] + "onnx"
        self.model = onnxruntime.InferenceSession(model_path)

    def predict(self, image: np.ndarray) -> str:
        """The model make prediction on image.

        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).

        Args:
            image (np.ndarray): Input image. Can be RGB/BGR or grayscale.

        Returns:
            str: Predicted text.
        """
        image = to_rgb(image=image)
        image = self.augmentor.resize_normalize(image=image)
        image = np.transpose(np.float32(image)[np.newaxis], (0, 3, 1, 2))

        ort_inputs = {self.model.get_inputs()[0].name: image}
        prediction = self.model.run(None, ort_inputs)

        prediction = self._prediction_to_text(prediction[0][0])
        return prediction


class InferenceDetONNX(BaseInferenceOCRDet):
    """Inference Detection model converted to ONNX."""

    def __init__(self, *args, **kwargs):
        """Inference OCR Detection model constructor with ONNX.

        Args:
            *args: Positional args for parent class.
            **kwargs: Keyword args for parent class.
        """
        super().__init__(*args, **kwargs)

    def _create_modules(self) -> None:
        """Create ONNX model."""
        self.visual = VisualDrawer(config=self.config)

        model_path = self.config["pretrained"][:-3] + "onnx"
        self.model = onnxruntime.InferenceSession(model_path)

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """The model make prediction on image.

        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).

        Args:
            image (np.ndarray): Input image. Can be RGB/BGR or grayscale.

        Returns:
            list of np.ndarray: List with predicted contours.
        """
        image = to_rgb(image=image)
        height, width = image.shape[:2]
        image = self.augmentor.resize_normalize(image=image, is_mask=False)
        image = np.transpose(np.float32(image)[np.newaxis], (0, 3, 1, 2))

        ort_inputs = {self.model.get_inputs()[0].name: image}
        prediction_mask = self.model.run(None, ort_inputs)

        prediction = self._find_contours(mask=prediction_mask[0][0])
        prediction = self.visual.prediction_to_original_size(prediction, height, width)
        return prediction
