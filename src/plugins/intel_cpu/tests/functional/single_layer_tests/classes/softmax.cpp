// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax.hpp"
#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string SoftMaxLayerCPUTest::getTestCaseName(const testing::TestParamInfo<softmaxCPUTestParams>& obj) {
    CPUSpecificParams cpuParams;
    ElementType inType;
    SoftMaxConfig config;
    std::string targetDevice;
    std::tie(inType, config, targetDevice, cpuParams) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << inType << "_";
    result << "IS=" << ov::test::utils::partialShape2str({config.inputShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : config.inputShape.second) {
        result << "(";
        result << ov::test::utils::vec2str(shape);
        result << ")_";
    }
    result << "axis=" << config.axis << "_";
    result << "trgDev=" << targetDevice;
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

void SoftMaxLayerCPUTest::SetUp() {
    ElementType inType;
    SoftMaxConfig config;
    CPUSpecificParams cpuParams;
    std::tie(inType, config, targetDevice, cpuParams) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }

    if (inType == ElementType::bf16) {
        rel_threshold = 2e-2f;
    }
    selectedType = makeSelectedTypeStr(selectedType, inType);
    init_input_shapes({config.inputShape});
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

    const auto softMax = std::make_shared<ov::op::v1::Softmax>(params.at(0), config.axis);

    function = makeNgraphFunction(inType, params, softMax, "SoftMax");
}

TEST_P(SoftMaxLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Softmax");
}

namespace SoftMax {

}  // namespace SoftMax
}  // namespace test
}  // namespace ov
