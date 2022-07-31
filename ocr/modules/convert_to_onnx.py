import torch
import onnxruntime

import numpy as np
from typing import List

from ocr.modules.load_config import load_config
from ocr_recognition.inference import InferenceOCRRec
from ocr_detection.inference import InferenceOCRDet


def to_numpy(tensor):
    """Convert tensor to numpy."""
    return tensor.detach().cpu().numpy() if \
        tensor.requires_grad else tensor.cpu().numpy()


def convert_to_onnx(model: torch.nn.Module, input_shape: List[int]):
    """Convert models to ONNX.

    Tested on Gyomei model type. Be careful, recognition models have few
    modules that may cause errors due conversion
    (f.e. Sequential modul).

    Args:
        model (torch.nn.Module): Pytorch-like model with loaded weights.
        input_shape (List[int]): Input data shape.
    """
    # Convert to ONNX
    with torch.no_grad():
        torch_inp = torch.randn(input_shape, requires_grad=False)
        torch_out = model(torch_inp)
        onnx_model = 'model.onnx'

        torch.onnx.export(model,
                          torch_inp,
                          onnx_model,
                          opset_version=11,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

    # Check conversion
    ort_session = onnxruntime.InferenceSession(onnx_model)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch_inp)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0],
                               rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, "
          "and the result looks good!")


def main():
    # ['ocr_detection', 'ocr_recognition']
    model_to_convert = 'detection'

    config = load_config(model_type=model_to_convert)
    config['device'] = 'cpu'

    if model_to_convert == 'recognition':
        model = InferenceOCRRec(config=config).model.model
    elif model_to_convert == 'detection':
        model = InferenceOCRDet(config=config).model.model
    else:
        AttributeError('Incorrect model name!')

    convert_to_onnx(model=model,
                    input_shape=[1, 3] + config['image_size'][::-1])


if __name__ == "__main__":
    main()
