// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "shape_of_inst.h"

#include <vector>
#include <iostream>

using namespace cldnn;
using namespace ::tests;

TEST(shape_of_gpu, bfyx) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{1, 2, 3, 3}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", input_info("input"), 4, data_types::i32));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> expected_results = {1, 2, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(shape_of_gpu, bfyx_i64) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{1, 2, 3, 3}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", input_info("input"), 4, data_types::i64));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    std::vector<int64_t> expected_results = {1, 2, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(shape_of_gpu, yxfb) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::yxfb, tensor{1, 2, 3, 3}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", input_info("input"), 4, data_types::i32));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> expected_results = {1, 2, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(shape_of_gpu, bfzyx) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfzyx, tensor{1, 2, 3, 3, 4}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", input_info("input"), 5, data_types::i32));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> expected_results = {1, 2, 4, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(shape_of_gpu, dynamic) {
    auto& engine = get_test_engine();

    layout in_layout = {ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    layout in_mem_layout0 = {ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx};
    layout in_mem_layout1 = {ov::PartialShape{4, 3, 2, 1}, data_types::f32, format::bfyx};
    auto input_mem0 = engine.allocate_memory(in_mem_layout0);
    auto input_mem1 = engine.allocate_memory(in_mem_layout1);

    cldnn::topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(shape_of("shape_of", input_info("input"), 5, data_types::i32));

    build_options bo;
    bo.set_option(build_option::allow_new_shape_infer(true));
    network network(engine, topology, bo);

    auto inst = network.get_primitive("shape_of");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    {
        network.set_input_data("input", input_mem0);

        auto outputs = network.execute();

        auto output = outputs.at("shape_of").get_memory();
        cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

        std::vector<int32_t> expected_results = {1, 2, 3, 4};

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i]));
        }
    }

    {
        network.set_input_data("input", input_mem1);

        auto outputs = network.execute();

        auto output = outputs.at("shape_of").get_memory();
        cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

        std::vector<int32_t> expected_results = {4, 3, 2, 1};

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i]));
        }
    }
}
