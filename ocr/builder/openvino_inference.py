import numpy as np
from typing import List
from openvino.inference_engine import IECore, Blob, TensorDesc

from ocr_recognition.visualizers.visualizer import to_rgb
from ocr_detection.visualizers.visualizer import VisualDrawer

from ocr_recognition.builder.base_inference import BaseInferenceOCRRec
from ocr_detection.builder.base_inference import BaseInferenceOCRDet


class InferenceRecOV(BaseInferenceOCRRec):
    """Inference Recognition model converted to OpenVINO.

    To convert use:
    mo --input_model model.onnx --input_shape [1,3,35,140]
    """

    def __init__(self, *args, **kwargs):
        """Inference OCR Recognition model constructor with OpenVINO.

        Args:
            *args: Positional args for parent class.
            **kwargs: Keyword args for parent class.
        """
        super().__init__(*args, **kwargs)

        self.tensor_description: TensorDesc = TensorDesc(
            precision="FP32",
            dims=([1, 3] + self.config["image_size"][::-1]),
            layout="NCHW"
        )

    def _create_modules(self) -> None:
        """Create OpenVINO model."""
        self._load_vocabulary()

        bin_path = self.config["pretrained"][:-3] + "bin"
        xml_path = self.config["pretrained"][:-3] + "xml"
        ie_core_handler = IECore()
        network = ie_core_handler.read_network(model=xml_path, weights=bin_path)
        self.model = ie_core_handler.load_network(network, device_name="CPU", num_requests=1)

    def predict(self, image: np.ndarray) -> str:
        """The model make prediction on image.

        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).

        Args:
            image (np.ndarray): Input image. Can be RGB/BGR or grayscale.

        Returns:
            str: Predicted word.
        """
        inference_request = self.model.requests[0]

        image = to_rgb(image=image)
        image = self.augmentor.resize_normalize(image=image)
        image = np.transpose(np.float32(image)[np.newaxis], (0, 3, 1, 2))

        # Create OpenVINO Blob and load it to Inference engine.
        input_blob = Blob(self.tensor_description, image)
        input_blob_name = next(iter(inference_request.input_blobs))
        inference_request.set_blob(blob_name=input_blob_name, blob=input_blob)

        # Run prediction.
        inference_request.infer()
        output_blob_name = next(iter(inference_request.output_blobs))
        prediction = inference_request.output_blobs[output_blob_name].buffer

        prediction = self._prediction_to_text(prediction[0])
        return prediction


class InferenceDetOV(BaseInferenceOCRDet):
    """Inference Detection model converted to OpenVINO.

    To convert use:
    mo --input_model model.onnx --input_shape [1,3,800,800]
    """

    def __init__(self, *args, **kwargs):
        """Inference OCR Detection model constructor with ONNX with OpenVINO.

        Args:
            *args: Positional args for parent class.
            **kwargs: Keyword args for parent class.
        """
        super().__init__(*args, **kwargs)

        self.tensor_description: TensorDesc = TensorDesc(
            precision="FP32",
            dims=([1, 3] + self.config["image_size"][::-1]),
            layout="NCHW"
        )

    def _create_modules(self) -> None:
        """Create OpenVINO model."""
        self.visual = VisualDrawer(config=self.config)

        bin_path = self.config["pretrained"][:-3] + "bin"
        xml_path = self.config["pretrained"][:-3] + "xml"
        ie_core_handler = IECore()
        network = ie_core_handler.read_network(model=xml_path, weights=bin_path)
        self.model = ie_core_handler.load_network(network, device_name="CPU", num_requests=1)

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """The model make prediction on image.

        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).

        Args:
            image (np.ndarray): Input image. Can be RGB/BGR
                or grayscale.

        Returns:
            list of np.ndarray: List with predicted contours.
        """
        inference_request = self.model.requests[0]

        image = to_rgb(image=image)
        height, width = image.shape[:2]
        image = self.augmentor.resize_normalize(image=image, is_mask=False)
        image = np.transpose(np.float32(image)[np.newaxis], (0, 3, 1, 2))

        # Create OpenVINO Blob and load it to Inference engine.
        input_blob = Blob(self.tensor_description, image)
        input_blob_name = next(iter(inference_request.input_blobs))
        inference_request.set_blob(blob_name=input_blob_name, blob=input_blob)

        # Run prediction.
        inference_request.infer()
        output_blob_name = next(iter(inference_request.output_blobs))
        prediction_mask = inference_request.output_blobs[output_blob_name].buffer

        prediction = self._find_contours(mask=prediction_mask[0])
        prediction = self.visual.prediction_to_original_size(prediction, height, width)
        return prediction
