from calendar import c
from math import exp
import time
import os
import pickle
import subprocess
import torch
import urllib3

import test_utils

TEST_RESOURCES_PATH = os.path.join(test_utils.REPO_ROOT, "cpp", "test", "resources", "torchscript_model", "mnist")
INPUT_DATA_PATH = os.path.join(TEST_RESOURCES_PATH, "0_png.pt")
MODEL_FILE_PATH = os.path.join(TEST_RESOURCES_PATH, "base_handler", "mnist_script.pt")

def setup_module(module):
    test_utils.torchserve_cleanup()
    if not os.path.exists(test_utils.MODEL_STORE):
        os.makedirs(test_utils.MODEL_STORE)

def teardown_module(module):
    test_utils.torchserve_cleanup()

def test_cpp_backend_mnist_base():

    model_archiver_cmd = test_utils.model_archiver_command_builder(
        model_name="mnist_base",
        version="1.0",
        serialized_file=MODEL_FILE_PATH,
        handler="BaseHandler",
        runtime="LSP",
        force=True
    )

    subprocess.run(model_archiver_cmd.split(" "))

    test_utils.start_torchserve(test_utils.MODEL_STORE, None, True, False)

    http = urllib3.PoolManager()

    load_url = "http://localhost:8081/models?url=mnist_base.mar&model_name=mnist_base&initial_workers=1"
    load_response = http.request("POST", load_url)
    time.sleep(2)

    inference_url = "http://localhost:8080/predictions/mnist_base"
    input_data_byte_array = bytearray()
    with open(INPUT_DATA_PATH, "rb") as img:
        input_data_byte_array += img.read()
    inference_response = http.request("GET", inference_url, body=input_data_byte_array)

    output_binary_file = os.path.join(test_utils.MODEL_STORE, "out.bin")
    with open(output_binary_file, "wb") as out_file:
        byte_array = bytearray(inference_response._body)
        out_file.write(byte_array)


    expected_result = torch.tensor([0.0000, -28.5285, -22.8017, -32.5117, -33.5584, -29.8429, -25.7716,
        -25.9097, -27.6592, -24.5729])

    computed_result = torch.load(output_binary_file)
    assert torch.allclose(computed_result, expected_result)
    test_utils.stop_torchserve()


if __name__ == "__main__":
    test_cpp_backend_mnist_base()

