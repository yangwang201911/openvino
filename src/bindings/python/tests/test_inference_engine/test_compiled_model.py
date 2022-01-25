# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

from ..conftest import model_path, read_image
from openvino.runtime import Model, ConstOutput, Shape

from openvino.runtime import Core, Tensor

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def test_get_metric(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    network_name = exec_net.get_metric("NETWORK_NAME")
    assert network_name == "test_model"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device dependent test")
def test_get_config(device):
    core = Core()
    if core.get_metric(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
        pytest.skip("Can't run on ARM plugin due-to CPU dependent test")
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    config = exec_net.get_config("PERF_COUNT")
    assert config == "NO"


def test_get_runtime_model(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    runtime_func = exec_net.get_runtime_model()
    assert isinstance(runtime_func, Model)


@pytest.mark.skip(reason="After infer will be implemented")
def test_export_import():
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, "CPU")
    exported_net_file = "exported_model.bin"
    exec_net.export_model(network_model=exported_net_file)
    assert os.path.exists(exported_net_file)
    exec_net = core.import_network(exported_net_file, "CPU")
    os.remove(exported_net_file)
    img = read_image()
    res = exec_net.infer({"data": img})
    assert np.argmax(res["fc_out"][0]) == 3
    del exec_net
    del core


def test_get_input_i(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    input = exec_net.input(0)
    input_node = input.get_node()
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_input_tensor_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    input = exec_net.input("data")
    input_node = input.get_node()
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_input(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    input = exec_net.input()
    input_node = input.get_node()
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_output_i(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    assert isinstance(output, ConstOutput)


def test_get_output(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output()
    assert isinstance(output, ConstOutput)


def test_input_set_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    input = exec_net.input("data")
    input_node = input.get_node()
    input_node.set_friendly_name("input_1")
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "input_1"


def test_output_set_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    output_node = output.get_node()
    output_node.set_friendly_name("output_1")
    name = output_node.friendly_name
    assert isinstance(output, ConstOutput)
    assert name == "output_1"


def test_outputs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    outputs = exec_net.outputs
    assert isinstance(outputs, list)
    assert len(outputs) == 1


def test_outputs_items(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    outputs = exec_net.outputs
    assert isinstance(outputs[0], ConstOutput)


def test_output_type(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    output_type = output.get_element_type().get_type_name()
    assert output_type == "f32"


def test_output_shape(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    expected_shape = Shape([1, 10])
    assert str(output.get_shape()) == str(expected_shape)


def test_input_get_index(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    input = exec_net.input(0)
    expected_idx = 0
    assert input.get_index() == expected_idx


def test_inputs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    assert isinstance(inputs, list)
    assert len(inputs) == 1


def test_inputs_items(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    assert isinstance(inputs[0], ConstOutput)


def test_inputs_get_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    name = node.friendly_name
    assert name == "data"


def test_inputs_set_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    node.set_friendly_name("input_0")
    name = node.friendly_name
    assert name == "input_0"


def test_inputs_docs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    expected_string = "openvino.runtime.ConstOutput wraps ov::Output<Const ov::Node >"
    assert input_0.__doc__ == expected_string


def test_infer_new_request_numpy(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = read_image()
    exec_net = ie.compile_model(func, device)
    res = exec_net.infer_new_request({"data": img})
    assert np.argmax(res[list(res)[0]]) == 2


def test_infer_new_request_tensor_numpy_copy(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = read_image()
    tensor = Tensor(img)
    exec_net = ie.compile_model(func, device)
    res_tensor = exec_net.infer_new_request({"data": tensor})
    res_img = exec_net.infer_new_request({"data": tensor})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 2
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_tensor_numpy_shared_memory(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = read_image()
    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    exec_net = ie.compile_model(func, device)
    res_tensor = exec_net.infer_new_request({"data": tensor})
    res_img = exec_net.infer_new_request({"data": tensor})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 2
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_new_request_wrong_port_name(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = read_image()
    tensor = Tensor(img)
    exec_net = ie.compile_model(func, device)
    with pytest.raises(KeyError) as e:
        exec_net.infer_new_request({"_data_": tensor})
    assert "Port for tensor named _data_ was not found!" in str(e.value)


def test_infer_tensor_wrong_input_data(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = read_image()
    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    exec_net = ie.compile_model(func, device)
    with pytest.raises(TypeError) as e:
        exec_net.infer_new_request({0.: tensor})
    assert "Incompatible key type for tensor named: 0." in str(e.value)


def test_infer_numpy_model_from_buffer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=bin)
    img = read_image()
    exec_net = core.compile_model(func, device)
    res = exec_net.infer_new_request({"data": img})
    assert np.argmax(res[list(res)[0]]) == 2


def test_infer_tensor_model_from_buffer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=bin)
    img = read_image()
    tensor = Tensor(img)
    exec_net = core.compile_model(func, device)
    res = exec_net.infer_new_request({"data": tensor})
    assert np.argmax(res[list(res)[0]]) == 2


def test_direct_infer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=bin)
    img = read_image()
    tensor = Tensor(img)
    comp_model = core.compile_model(model, device)
    res = comp_model({"data": tensor})
    assert np.argmax(res[comp_model.outputs[0]]) == 2
    ref = comp_model.infer_new_request({"data": tensor})
    assert np.array_equal(ref[comp_model.outputs[0]], res[comp_model.outputs[0]])
