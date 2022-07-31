from openvino.inference_engine import IECore, Blob, TensorDesc
import numpy as np


def main():
    # to convert onnx run followinf command:
    # mo --input_model best_Model.onnx --input_shape [1,3,800,800]
    mode = 'recognition'
    bin_path = f'./ocr/data/ocr_{mode}/model.bin'
    xml_path = f'./ocr/data/ocr_{mode}/model.xml'

    ie_core_handler = IECore()
    network = ie_core_handler.read_network(model=xml_path,
                                           weights=bin_path)

    executable_network = ie_core_handler.load_network(network,
                                                      device_name='CPU',
                                                      num_requests=1)

    inference_request = executable_network.requests[0]

    random_input_data = np.random.randn(1, 3, 35, 140).astype(
        np.float32)
    tensor_description = TensorDesc(precision="FP32",
                                    dims=(1, 3, 35, 140), layout='NCHW')
    Blob(tensor_description, random_input_data)

    print(inference_request.input_blobs)

    next(iter(inference_request.input_blobs))
    inference_request.infer()
    output_blob_name = next(iter(inference_request.output_blobs))
    output = inference_request.output_blobs[output_blob_name].buffer
    print(output.shape)


if __name__ == '__main__':
    main()
