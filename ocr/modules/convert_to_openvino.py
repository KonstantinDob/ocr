from openvino.inference_engine import IECore, Blob, TensorDesc
import numpy as np

# mo --input_model best_Model.onnx --input_shape [1,3,800,800]
BIN_PATH = '/home/konstantin/workdir/ocr/data/ocr_detection/models/model.bin'
XML_PATH = '/home/konstantin/workdir/ocr/data/ocr_detection/models/model.xml'

ie_core_handler = IECore()
network = ie_core_handler.read_network(model=XML_PATH, weights=BIN_PATH)

executable_network = ie_core_handler.load_network(network, device_name='CPU', num_requests=1)

inference_request = executable_network.requests[0]

random_input_data = np.random.randn(1, 3, 35, 140).astype(np.float32)
tensor_description = TensorDesc(precision="FP32", dims=(1, 3, 35, 140), layout='NCHW')
input_blob = Blob(tensor_description, random_input_data)

print(inference_request.input_blobs)

input_blob_name = next(iter(inference_request.input_blobs))
inference_request.infer()
output_blob_name = next(iter(inference_request.output_blobs))
output = inference_request.output_blobs[output_blob_name].buffer
print(output.shape)