from calendar import c
from math import exp
import time
import os
import pickle
import subprocess
import torch
import urllib3

import test_utils

MODEL_STORE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "cpp")
print(MODEL_STORE_PATH)



def test_cpp_backend_mnist_base():
    # TODO: generate mar file
    test_utils.start_torchserve(MODEL_STORE_PATH, None, True, False)
    http = urllib3.PoolManager()

    load_url = "http://localhost:8081/models?url=mnist_base.mar&model_name=mnist_base&initial_workers=1"
    load_response = http.request("POST", load_url)
    time.sleep(2)

    inference_url = "http://localhost:8080/predictions/mnist_base"
    input_data_byte_array = bytearray()
    with open(os.path.join(MODEL_STORE_PATH, "0_png.pt"), "rb") as img:
        input_data_byte_array += img.read()
    inference_response = http.request("GET", inference_url, body=input_data_byte_array)

    out_pickle_path = os.path.join(MODEL_STORE_PATH, "out.pkl")
    with open(out_pickle_path, "wb") as out_file:
        byte_array = bytearray(inference_response._body)
        out_file.write(byte_array)


    expected_result = torch.tensor([0.0000, -28.5285, -22.8017, -32.5117, -33.5584, -29.8429, -25.7716,
        -25.9097, -27.6592, -24.5729])
    
    computed_result = torch.load(out_pickle_path)
    assert torch.allclose(computed_result, expected_result)
    test_utils.stop_torchserve()


if __name__ == "__main__":
    test_cpp_backend_mnist_base()

