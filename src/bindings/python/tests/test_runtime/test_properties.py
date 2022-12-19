# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import os

from openvino.runtime import Core, Type, OVAny, properties


###
# Base properties API
###
def test_properties_ro_base():
    with pytest.raises(TypeError) as e:
        properties.supported_properties("something")
    assert "incompatible function arguments" in str(e.value)


def test_properties_rw_base():
    assert properties.cache_dir() == "CACHE_DIR"
    assert properties.cache_dir("./test_dir") == ("CACHE_DIR", OVAny("./test_dir"))

    with pytest.raises(TypeError) as e:
        properties.cache_dir(6)
    assert "incompatible function arguments" in str(e.value)


###
# Enum-like values
###
@pytest.mark.parametrize(
    ("ov_enum", "expected_values"),
    [
        (
            properties.Affinity,
            (
                (properties.Affinity.NONE, "Affinity.NONE", -1),
                (properties.Affinity.CORE, "Affinity.CORE", 0),
                (properties.Affinity.NUMA, "Affinity.NUMA", 1),
                (properties.Affinity.HYBRID_AWARE, "Affinity.HYBRID_AWARE", 2),
            ),
        ),
        (
            properties.hint.Priority,
            (
                (properties.hint.Priority.LOW, "Priority.LOW", 0),
                (properties.hint.Priority.MEDIUM, "Priority.MEDIUM", 1),
                (properties.hint.Priority.HIGH, "Priority.HIGH", 2),
                (properties.hint.Priority.DEFAULT, "Priority.MEDIUM", 1),
            ),
        ),
        (
            properties.hint.PerformanceMode,
            (
                (properties.hint.PerformanceMode.UNDEFINED, "PerformanceMode.UNDEFINED", -1),
                (properties.hint.PerformanceMode.LATENCY, "PerformanceMode.LATENCY", 1),
                (properties.hint.PerformanceMode.THROUGHPUT, "PerformanceMode.THROUGHPUT", 2),
                (properties.hint.PerformanceMode.CUMULATIVE_THROUGHPUT, "PerformanceMode.CUMULATIVE_THROUGHPUT", 3),
            ),
        ),
        (
            properties.device.Type,
            (
                (properties.device.Type.INTEGRATED, "Type.INTEGRATED", 0),
                (properties.device.Type.DISCRETE, "Type.DISCRETE", 1),
            ),
        ),
        (
            properties.log.Level,
            (
                (properties.log.Level.NO, "Level.NO", -1),
                (properties.log.Level.ERR, "Level.ERR", 0),
                (properties.log.Level.WARNING, "Level.WARNING", 1),
                (properties.log.Level.INFO, "Level.INFO", 2),
                (properties.log.Level.DEBUG, "Level.DEBUG", 3),
                (properties.log.Level.TRACE, "Level.TRACE", 4),
            ),
        ),
    ],
)
def test_properties_enums(ov_enum, expected_values):
    assert ov_enum is not None
    enum_entries = iter(ov_enum.__entries.values())

    for property_obj, property_str, property_int in expected_values:
        assert property_obj == next(enum_entries)[0]
        assert str(property_obj) == property_str
        assert int(property_obj) == property_int


###
# Read-Only properties
###
@pytest.mark.parametrize(
    ("ov_property_ro", "expected_value"),
    [
        (properties.supported_properties, "SUPPORTED_PROPERTIES"),
        (properties.available_devices, "AVAILABLE_DEVICES"),
        (properties.model_name, "NETWORK_NAME"),
        (properties.optimal_number_of_infer_requests, "OPTIMAL_NUMBER_OF_INFER_REQUESTS"),
        (properties.range_for_streams, "RANGE_FOR_STREAMS"),
        (properties.optimal_batch_size, "OPTIMAL_BATCH_SIZE"),
        (properties.max_batch_size, "MAX_BATCH_SIZE"),
        (properties.range_for_async_infer_requests, "RANGE_FOR_ASYNC_INFER_REQUESTS"),
        (properties.device.full_name, "FULL_DEVICE_NAME"),
        (properties.device.architecture, "DEVICE_ARCHITECTURE"),
        (properties.device.type, "DEVICE_TYPE"),
        (properties.device.gops, "DEVICE_GOPS"),
        (properties.device.thermal, "DEVICE_THERMAL"),
        (properties.device.capabilities, "OPTIMIZATION_CAPABILITIES"),
    ],
)
def test_properties_ro(ov_property_ro, expected_value):
    # Test if property is correctly registered
    assert ov_property_ro() == expected_value


###
# Read-Write properties
###
@pytest.mark.parametrize(
    ("ov_property_rw", "expected_value", "test_values"),
    [
        (
            properties.enable_profiling,
            "PERF_COUNT",
            (
                (True, True),
                (False, False),
                (1, True),
                (0, False),
            ),
        ),
        (
            properties.cache_dir,
            "CACHE_DIR",
            (("./test_cache", "./test_cache"),),
        ),
        (
            properties.auto_batch_timeout,
            "AUTO_BATCH_TIMEOUT",
            (
                (21, 21),
                (np.uint32(37), 37),
                (21, np.uint32(21)),
                (np.uint32(37), np.uint32(37)),
            ),
        ),
        (
            properties.inference_num_threads,
            "INFERENCE_NUM_THREADS",
            (
                (-8, -8),
                (8, 8),
            ),
        ),
        (
            properties.compilation_num_threads,
            "COMPILATION_NUM_THREADS",
            ((44, 44),),
        ),
        (
            properties.affinity,
            "AFFINITY",
            ((properties.Affinity.NONE, properties.Affinity.NONE),),
        ),
        (properties.force_tbb_terminate, "FORCE_TBB_TERMINATE", ((True, True),)),
        (properties.hint.inference_precision, "INFERENCE_PRECISION_HINT", ((Type.f32, Type.f32),)),
        (
            properties.hint.model_priority,
            "MODEL_PRIORITY",
            ((properties.hint.Priority.LOW, properties.hint.Priority.LOW),),
        ),
        (
            properties.hint.performance_mode,
            "PERFORMANCE_HINT",
            ((properties.hint.PerformanceMode.UNDEFINED, properties.hint.PerformanceMode.UNDEFINED),),
        ),
        (
            properties.hint.num_requests,
            "PERFORMANCE_HINT_NUM_REQUESTS",
            ((8, 8),),
        ),
        (
            properties.hint.allow_auto_batching,
            "ALLOW_AUTO_BATCHING",
            ((True, True),),
        ),
        (
            properties.intel_cpu.denormals_optimization,
            "CPU_DENORMALS_OPTIMIZATION",
            ((True, True),),
        ),
        (
            properties.intel_cpu.sparse_weights_decompression_rate,
            "SPARSE_WEIGHTS_DECOMPRESSION_RATE",
            (
                (0.1, np.float32(0.1)),
                (2.0, 2.0),
            ),
        ),
        (properties.device.id, "DEVICE_ID", (("0", "0"),)),
        (
            properties.log.level,
            "LOG_LEVEL",
            ((properties.log.Level.NO, properties.log.Level.NO),),
        ),
    ],
)
def test_properties_rw(ov_property_rw, expected_value, test_values):
    # Test if property is correctly registered
    assert ov_property_rw() == expected_value

    # Test if property process values correctly
    for values in test_values:
        property_tuple = ov_property_rw(values[0])
        assert property_tuple[0] == expected_value
        assert property_tuple[1].value == values[1]


###
# Special cases
###
def test_properties_device_priorities():
    assert properties.device.priorities() == "MULTI_DEVICE_PRIORITIES"
    assert properties.device.priorities("CPU,GPU") == ("MULTI_DEVICE_PRIORITIES", OVAny("CPU,GPU,"))
    assert properties.device.priorities("CPU", "GPU") == ("MULTI_DEVICE_PRIORITIES", OVAny("CPU,GPU,"))

    with pytest.raises(TypeError) as e:
        value = 6
        properties.device.priorities("CPU", value)
    assert f"Incorrect passed value: {value} , expected string values." in str(e.value)


def test_properties_streams():
    # Test extra Num class
    assert properties.streams.Num().to_integer() == -1
    assert properties.streams.Num(2).to_integer() == 2
    assert properties.streams.Num.AUTO.to_integer() == -1
    assert properties.streams.Num.NUMA.to_integer() == -2
    # Test RW property
    property_tuple = properties.streams.num(properties.streams.Num.AUTO)
    assert property_tuple[0] == "NUM_STREAMS"
    assert property_tuple[1].value == -1

    property_tuple = properties.streams.num(42)
    assert property_tuple[0] == "NUM_STREAMS"
    assert property_tuple[1].value == 42


def test_properties_capability():
    assert properties.device.Capability.FP32 == "FP32"
    assert properties.device.Capability.BF16 == "BF16"
    assert properties.device.Capability.FP16 == "FP16"
    assert properties.device.Capability.INT8 == "INT8"
    assert properties.device.Capability.INT16 == "INT16"
    assert properties.device.Capability.BIN == "BIN"
    assert properties.device.Capability.WINOGRAD == "WINOGRAD"
    assert properties.device.Capability.EXPORT_IMPORT == "EXPORT_IMPORT"


def test_properties_hint_model():
    # Temporary imports
    from tests.test_utils.test_utils import generate_add_model

    model = generate_add_model()

    assert properties.hint.model() == "MODEL_PTR"

    property_tuple = properties.hint.model(model)
    assert property_tuple[0] == "MODEL_PTR"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_single_property_setting():
    core = Core()

    if "Intel" not in core.get_property("CPU", "FULL_DEVICE_NAME"):
        pytest.skip("This test runs only on openvino intel cpu plugin")

    core.set_property("CPU", properties.streams.num(properties.streams.Num.AUTO))

    assert properties.streams.Num.AUTO.to_integer() == -1
    assert type(core.get_property("CPU", properties.streams.num())) == int


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
@pytest.mark.parametrize(
    "properties_to_set",
    [
        # Dict from list of tuples
        dict(  # noqa: C406
            [  # noqa: C406
                properties.enable_profiling(True),
                properties.cache_dir("./"),
                properties.inference_num_threads(9),
                properties.affinity(properties.Affinity.NONE),
                properties.hint.inference_precision(Type.f32),
                properties.hint.performance_mode(properties.hint.PerformanceMode.LATENCY),
                properties.hint.num_requests(12),
                properties.streams.num(5),
            ],
        ),
        # Pure dict
        {
            properties.enable_profiling(): True,
            properties.cache_dir(): "./",
            properties.inference_num_threads(): 9,
            properties.affinity(): properties.Affinity.NONE,
            properties.hint.inference_precision(): Type.f32,
            properties.hint.performance_mode(): properties.hint.PerformanceMode.LATENCY,
            properties.hint.num_requests(): 12,
            properties.streams.num(): 5,
        },
        # Mixed dict
        {
            properties.enable_profiling(): True,
            "CACHE_DIR": "./",
            properties.inference_num_threads(): 9,
            properties.affinity(): "NONE",
            "INFERENCE_PRECISION_HINT": Type.f32,
            properties.hint.performance_mode(): properties.hint.PerformanceMode.LATENCY,
            properties.hint.num_requests(): 12,
            "NUM_STREAMS": properties.streams.Num(5),
        },
    ],
)
def test_properties_core(properties_to_set):
    core = Core()

    if "Intel" not in core.get_property("CPU", "FULL_DEVICE_NAME"):
        pytest.skip("This test runs only on openvino intel cpu plugin")

    core.set_property(properties_to_set)

    # RW properties without device name
    assert core.get_property(properties.cache_dir()) == "./"
    assert core.get_property(properties.force_tbb_terminate()) is False

    # RW properties
    assert core.get_property("CPU", properties.enable_profiling()) is True
    assert core.get_property("CPU", properties.cache_dir()) == "./"
    assert core.get_property("CPU", properties.inference_num_threads()) == 9
    assert core.get_property("CPU", properties.affinity()) == properties.Affinity.NONE
    assert core.get_property("CPU", properties.hint.inference_precision()) == Type.f32
    assert core.get_property("CPU", properties.hint.performance_mode()) == properties.hint.PerformanceMode.LATENCY
    assert core.get_property("CPU", properties.hint.num_requests()) == 12
    assert core.get_property("CPU", properties.streams.num()) == 5

    # RO properties
    assert type(core.get_property("CPU", properties.supported_properties())) == dict
    assert type(core.get_property("CPU", properties.available_devices())) == list
    assert type(core.get_property("CPU", properties.optimal_number_of_infer_requests())) == int
    assert type(core.get_property("CPU", properties.range_for_streams())) == tuple
    assert type(core.get_property("CPU", properties.range_for_async_infer_requests())) == tuple
    assert type(core.get_property("CPU", properties.device.full_name())) == str
    assert type(core.get_property("CPU", properties.device.capabilities())) == list
