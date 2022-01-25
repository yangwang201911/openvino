// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils/summary.hpp"

namespace ov {
namespace test {

using InputShape = std::pair<ov::PartialShape, std::vector<ov::Shape>>;
std::ostream& operator <<(std::ostream& os, const InputShape& inputShape);

using ElementType = ov::element::Type_t;
using Config = std::map<std::string, std::string>;
using TargetDevice = std::string;

class SubgraphBaseTest : public CommonTestUtils::TestsCommon {
public:
    virtual void run();
    virtual void serialize();
    virtual void query_model();

    void TearDown() override {
        if (!configuration.empty()) {
            ov::test::utils::PluginCache::get().core().reset();
        }
    }

protected:
    virtual void compare(const std::vector<ov::Tensor> &expected,
                         const std::vector<ov::Tensor> &actual);

    virtual void configure_model();
    virtual void compile_model();
    virtual void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes);
    virtual void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes);
    virtual void infer();
    virtual void validate();

    void init_input_shapes(const std::vector<InputShape>& shapes);

    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::string targetDevice;
    Config configuration;

    std::shared_ptr<ov::Model> function, functionRefs = nullptr;
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs;
    std::vector<ov::PartialShape> inputDynamicShapes;
    std::vector<std::vector<ov::Shape>> targetStaticShapes;
    ElementType inType = ov::element::undefined, outType = ov::element::undefined;

    ov::CompiledModel executableNetwork;
    ov::InferRequest inferRequest;

    constexpr static const double disable_threshold = std::numeric_limits<double>::max();
    double abs_threshold = disable_threshold, rel_threshold = disable_threshold;

    LayerTestsUtils::Summary& summary = LayerTestsUtils::Summary::getInstance();

    virtual std::vector<ov::Tensor> calculate_refs();
    virtual std::vector<ov::Tensor> get_plugin_outputs();
};

inline std::vector<std::vector<InputShape>> static_shapes_to_test_representation(const std::vector<std::vector<ov::Shape>>& shapes) {
    std::vector<std::vector<InputShape>> result;
    for (const auto& staticShapes : shapes) {
        std::vector<InputShape> tmp;
        for (const auto& staticShape : staticShapes) {
            tmp.push_back({{}, {staticShape}});
        }
        result.push_back(tmp);
    }
    return result;
}

inline std::vector<InputShape> static_shapes_to_test_representation(const std::vector<ov::Shape>& shapes) {
    std::vector<InputShape> result;
    for (const auto& staticShape : shapes) {
        result.push_back({{}, {staticShape}});
    }
    return result;
}
}  // namespace test
}  // namespace ov
