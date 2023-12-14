// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <common/blocked_desc_creator.h>
#include <cpu_types.h>
#include <edge.h>
#include <gtest/gtest.h>
#include <ie_common.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include <memory_desc/dnnl_memory_desc.h>
#include <node.h>
#include <nodes/reorder.h>

#include <common/memory_desc_wrapper.hpp>
#include <dnnl.hpp>
#include <utility>

#include "common_test_utils/common_utils.hpp"
#include "cache/multi_cache.h"
#include "ov_models/builders.hpp"
#include "nodes/scaled_attn.h"
#include "nodes/input.h"
#include "nodes/convert.h"
#include "graph.h"
#include "cpu_tensor.h"

using namespace ov::intel_cpu;

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 126975
TEST(ScaledAttnGraphTest, DISABLED_smoke_Check_Scaled_Concat_Noplace) {
#else
TEST(ScaledAttnGraphTest, smoke_Check_Scaled_Concat_Noplace) {
#endif
    auto build_graph = [](const ov::Shape& shape, float* qkv_val, float* past_kv_val) {
        auto qkv = ov::op::v0::Constant::create(ov::element::f32, shape, qkv_val);
        qkv->set_friendly_name("qkv_const");
        auto pastkv_f32 = ov::op::v0::Constant::create(ov::element::f32, shape, past_kv_val);
        pastkv_f32->set_friendly_name("pastkv_const_f32");
        auto pastkv = std::make_shared<ov::op::v0::Convert>(pastkv_f32, ov::element::f16);
        pastkv->set_friendly_name("pastkv_const");
        // only need a dynamic parameter but its value will not be used
        auto attn = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
        attn->set_friendly_name("attn");

        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;
        config.fuse_concat = true;
        config.is_causal = true;
        auto sdpa = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(ov::OutputVector{qkv, qkv, qkv, attn, pastkv, pastkv}, config);
        auto out_pastk_convert = std::make_shared<ov::op::v0::Convert>(sdpa->output(1), ov::element::f32);
        auto out_pastv_convert = std::make_shared<ov::op::v0::Convert>(sdpa->output(2), ov::element::f32);
        auto out_qkv = std::make_shared<ov::op::v0::Result>(sdpa->output(0));
        out_qkv->set_friendly_name("qkv");
        auto out_pastk = std::make_shared<ov::op::v0::Result>(out_pastk_convert);
        out_pastk->set_friendly_name("pastk");
        auto out_pastv = std::make_shared<ov::op::v0::Result>(out_pastv_convert);
        out_pastv->set_friendly_name("pastv");

        std::unordered_set<NodePtr> nodes_set;
        std::vector<EdgePtr> graph_edges;

        auto add_edge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
            auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
            child->addEdge(edge);
            graph_edges.push_back(edge);
            nodes_set.insert(parent);
            nodes_set.insert(child);
        };

        //create graph context
        Config conf;
        conf.rtCacheCapacity = 0;
        auto context = std::make_shared<GraphContext>(conf, nullptr, nullptr, false);

        auto qkv_node = std::make_shared<node::Input>(qkv, context);
        auto pastkv_f32_node = std::make_shared<node::Input>(pastkv_f32, context);
        auto attn_node = std::make_shared<node::Input>(attn, context);
        auto pastkv_node = std::make_shared<node::Convert>(pastkv, context);
        auto sdpa_node = std::make_shared<node::ScaledDotProductAttention>(sdpa, context);
        auto out_pastk_node_convert = std::make_shared<node::Convert>(out_pastk_convert, context);
        auto out_pastv_node_convert = std::make_shared<node::Convert>(out_pastv_convert, context);
        auto out_qkv_node = std::make_shared<node::Input>(out_qkv, context);
        auto out_pastk_node = std::make_shared<node::Input>(out_pastk, context);
        auto out_pastv_node = std::make_shared<node::Input>(out_pastv, context);

        add_edge(qkv_node, sdpa_node, 0, 0);
        add_edge(qkv_node, sdpa_node, 0, 1);
        add_edge(qkv_node, sdpa_node, 0, 2);
        add_edge(attn_node, sdpa_node, 0, 3);
        add_edge(pastkv_f32_node, pastkv_node, 0, 0);
        add_edge(pastkv_node, sdpa_node, 0, 4);
        add_edge(pastkv_node, sdpa_node, 0, 5);
        add_edge(sdpa_node, out_qkv_node, 0, 0);
        add_edge(sdpa_node, out_pastk_node_convert, 1, 0);
        add_edge(sdpa_node, out_pastv_node_convert, 2, 0);
        add_edge(out_pastk_node_convert, out_pastk_node, 0, 0);
        add_edge(out_pastv_node_convert, out_pastv_node, 0, 0);

        std::vector<NodePtr> graph_nodes(nodes_set.begin(), nodes_set.end());

        Graph graph;
        graph.CreateGraph(graph_nodes, graph_edges, context, "test_graph");
        return graph;
    };

    auto run_graph = [] (Graph& graph) {
        graph.GetInputNodesMap().begin()->second->redefineOutputMemory(0, {1});

        for (auto& node : graph.GetNodes()) {
            if (node->isDynamicNode()) {
                node->updateShapes();
                node->updateDynamicParams();
            }
        }
        graph.Infer();
    };

    auto check_graph = [] (Graph& graph, std::map<std::string, std::pair<float*, ov::Shape>>& expected) {
        auto& outputNodesMap = graph.GetOutputNodesMap();
        auto is_same = [] (float a, float b) {
            return std::abs(a - b) < 0.01f;
        };
        for (auto &outputMap : outputNodesMap) {
            auto name = outputMap.first;
            if (expected.count(name) == 0) {
                continue;
            }
            auto node = outputMap.second;
            auto parentEdge = node->getParentEdgeAt(0);
            const auto& memory = parentEdge->getMemoryPtr();
            auto size = memory->getSize() / sizeof(float);
            auto p = reinterpret_cast<float*>(memory->getData());
            for (size_t i = 0; i < size; i++) {
                ASSERT_EQ(is_same(p[i], expected.at(name).first[i]), true);
            }
            ASSERT_EQ(memory->getShape(), ov::intel_cpu::Shape(expected.at(name).second));
        }
    };

    auto find_node_type = [](const Graph& graph, Type type) -> NodePtr {
        auto&& nodes = graph.GetNodes();
        auto itr =
            std::find_if(nodes.begin(), nodes.end(), [=](const NodePtr& node){ return type == node->getType(); });

        if (itr == nodes.end()) {
            return nullptr;
        }

        return (*itr);
    };

    auto strided_iota = [] (float* first, size_t n, float value, float stride) {
        for (size_t i = 0; i < n; i++) {
            *first++ = value;
            value += stride;
        }
    };

    ov::Shape shape{1, 1, 8, 8};
    const size_t elements_count = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<float> val(elements_count * 2);
    strided_iota(val.data(), val.size(), -10.0f, 0.1f);
    auto graph = build_graph(shape, val.data() + elements_count, val.data());
    run_graph(graph);
    // if no inplace, the pastk and pastv will concat, check shape and value
    ov::Shape expectedShape(shape);
    expectedShape[2] *= 2;
    std::map<std::string, std::pair<float*, ov::Shape>> expected{
        {"pastk", std::make_pair(val.data(), expectedShape)},
        {"pastv", std::make_pair(val.data(), expectedShape)}};
    check_graph(graph, expected);
    auto spd = find_node_type(graph, Type::ScaledDotProductAttention)->getSelectedPrimitiveDescriptor();
    ASSERT_EQ(spd->getConfig().outConfs[1].inPlace(), -1);
    ASSERT_EQ(spd->getConfig().outConfs[2].inPlace(), -1);
}