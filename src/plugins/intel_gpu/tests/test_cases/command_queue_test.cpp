// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "test_utils/test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/arg_max_min.hpp>

using namespace cldnn;
using namespace ::tests;
using namespace std;

namespace {
// Run some topology too see if command queue does work correctly
// Coppied from arg_max_gpu.base test.
void exexute_network(cldnn::engine& engine, bool is_caching_test=false) {
    //  Input  : 2x4x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size, y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { input_info("input") }, ov::op::TopKMode::MIN, top_k, 0));

    std::vector<float> input_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   10.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f3*/4.f,  0.5f,  8.f,   8.2f
    };
    set_values(input, input_vec);

    cldnn::network::ptr network;

    if (is_caching_test) {
        membuf mem_buf;
        {
            cldnn::network _network(engine, topology);
            std::ostream out_mem(&mem_buf);
            BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
            _network.save(ob);
        }
        {
            std::istream in_mem(&mem_buf);
            BinaryInputBuffer ib = BinaryInputBuffer(in_mem, engine);
            network = std::make_shared<cldnn::network>(ib, get_test_stream_ptr(), engine);
        }
    } else {
        network = std::make_shared<cldnn::network>(engine, topology);
    }

    network->set_input_data("input", input);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
    }
}
}  // namespace

TEST(command_queue_test, test_priority_hints) {
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            queue_types::out_of_order,
            "",             // sources_dumps_dir
            priority_mode_types::low,
            throttle_mode_types::disabled);
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, configuration);
    exexute_network(*engine);
}

TEST(command_queue_test, test_throttle_hints) {
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            queue_types::out_of_order,
            "",             // sources_dumps_dir
            priority_mode_types::disabled,
            throttle_mode_types::high);
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, configuration);
    exexute_network(*engine);
}

TEST(command_queue_test, test_priority_and_throttle_hints) {
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            queue_types::out_of_order,
            "",             // sources_dumps_dir
            priority_mode_types::high,
            throttle_mode_types::low);
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, configuration);
    exexute_network(*engine);
}

TEST(export_import_command_queue_test, test_priority_and_throttle_hints) {
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            queue_types::out_of_order,
            "",             // sources_dumps_dir
            priority_mode_types::high,
            throttle_mode_types::low);
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, configuration);
    exexute_network(*engine, true);
}
