// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <shared_test_classes/single_layer/convolution.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
using LayerTestsDefinitions::convSpecificParams;

typedef std::tuple<
        convSpecificParams,
        ElementType,     // Net precision
        ElementType,     // Input precision
        ElementType,     // Output precision
        InputShape,      // Input shape
        LayerTestsUtils::TargetDevice   // Device name
> convLayerTestParamsSet;

typedef std::tuple<
        convLayerTestParamsSet,
        CPUSpecificParams,
        fusingSpecificParams,
        std::map<std::string, std::string> > convLayerCPUTestParamsSet;

class ConvolutionLayerCPUTest : public testing::WithParamInterface<convLayerCPUTestParamsSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerCPUTestParamsSet>& obj) {
        convLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

        convSpecificParams convParams;
        ElementType netType;
        ElementType inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = basicParamsSet;
        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }
protected:
    bool isBias = false;

    void checkBiasFusing(ov::CompiledModel &execNet) const {
        auto execGraph = execNet.get_runtime_model();
        ASSERT_NE(nullptr, execGraph);

        bool foundConv = false;
        for (const auto &node : execGraph->get_ops()) {
            const auto & rtInfo = node->get_rt_info();
            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                IE_ASSERT(rtInfo.end() != it);
                return it->second.as<std::string>();
            };

            if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Convolution") {
                foundConv = true;
                ASSERT_EQ(3, node->inputs().size());
                break;
            }
        }

        ASSERT_TRUE(foundConv) << "Can't find Convolution node";
    }

    std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                              ngraph::ParameterVector &params,
                                              const std::shared_ptr<ngraph::Node> &lastNode) override {
        auto retNode = CpuTestWithFusing::modifyGraph(ngPrc, params, lastNode);
        for (size_t i = targetStaticShapes.front().size(); i < params.size(); ++i) {
            const auto& shape = params[i]->get_output_partial_shape(0);
            if (shape.is_static()) {
                targetStaticShapes.front().push_back(shape.get_shape());
            }
        }
        return retNode;
    }

    void SetUp() override {
        rel_threshold = 1e-4f;

        convLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        if (postOpMgrPtr)
            isBias = (postOpMgrPtr->getFusedOpsNames() == "Add(PerChannel)" && selectedType != "jit_avx512_winograd");

        convSpecificParams convParams;
        InputShape inputShape;
        auto netType = ElementType::undefined;
        std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = basicParamsSet;

        init_input_shapes({inputShape});

        if (configuration.count(PluginConfigParams::KEY_ENFORCE_BF16) &&
                PluginConfigParams::YES == configuration[PluginConfigParams::KEY_ENFORCE_BF16]) {
            selectedType += "_BF16";
            rel_threshold = 1e-2f;
            if (selectedType == "jit_gemm_BF16")
                rel_threshold = 0.05f;
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netType);
        }

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        auto inputParams = ngraph::builder::makeDynamicParams(ngraph::element::f32, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

        auto convolutionNode = ngraph::builder::makeConvolution(paramOuts.front(), netType, kernel, stride, padBegin,
                                                                padEnd, dilation, padType, convOutChannels);

        function = makeNgraphFunction(netType, inputParams, convolutionNode, "Convolution");
    }
};

TEST_P(ConvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    // Skip tests for sse41 convolution where ic or oc cannot be exactly divided by the block size,
    // since tails processing for sse41 nspc layout is not supported yet (see 52736).
    if (!inFmts.empty() && (inFmts.front() == nwc || inFmts.front() == nhwc || inFmts.front() == ndhwc) && selectedType.find("jit_sse") != std::string::npos) {
        auto inpChannels = function->get_parameters().front()->get_partial_shape()[1].get_length();
        auto outChannels = function->get_output_partial_shape(0)[1].get_length();
        if ((inpChannels % 8) || (outChannels % 8)) {
            GTEST_SKIP() << "Disabled test due to the sse41 convolution kernel does not support tails for nspc layout." << std::endl;
        }
    }

    run();

    if (isBias) {
        checkBiasFusing(executableNetwork);
    }
    CheckPluginRelatedResults(executableNetwork, "Convolution");
}

namespace {

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        // eltwise
        fusingRelu,
        fusingPRelu1D,
        // depthwise
        fusingReluScaleShift,
        // fake quantize
        fusingFakeQuantizePerTensorRelu,
        fusingFakeQuantizePerChannelRelu,
        // sum
        fusingSumEluFQ,
        fusingSum,
        // bias
        fusingAddPerChannel
};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{
        emptyFusingSpec,
        // eltwise
        fusingRelu,
        // depthwise
        fusingReluScaleShift,
        // sum
        fusingSum,
        // bias
        fusingAddPerChannel
};

/* ============= Convolution params (GEMM layout) ============= */
const SizeVector numOutChannels_Gemm = { 6 };

/* ============= Convolution params (blocked and nspc layout) ============= */
const SizeVector numOutChannels = { 64, 63 };

/* ============= Convolution params (1D) ============= */
const std::vector<SizeVector> kernels1d = { {3}, {1} };
const std::vector<SizeVector> strides1d = { {1}, {2} };
const std::vector<std::vector<ptrdiff_t>> padBegins1d = { {0}, {1} };
const std::vector<std::vector<ptrdiff_t>> padEnds1d = { {0} };
const std::vector<SizeVector> dilations1d = { {1}, {2} };
std::vector<InputShape> inputShapes1d = {
        {{}, {{ 2, 64, 7 }}},
        {{}, {{ 1, 67, 7 }}},
        {
            //dynamic shape
            { -1, 64, {1, 200} },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 9 }
            }
        },
        {
            //dynamic shape
            { -1, 67, {1, 200} },
            { //target static shapes
                { 2, 67, 7 },
                { 1, 67, 9 }
            }
        },
        {
            //dynamic shape
            { {1, 200}, 64, -1 },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 5 }
            }
        }
};
std::vector<InputShape> inputShapesPlain2Blocked1d = {
        {{}, {{1, 1, 7}}},
        {{}, {{1, 2, 7}}},
        {{}, {{1, 3, 7}}},
        {
        //dynamic shapes
            {-1, 1, {1, 200}},
            { //target static shapes
                {2, 1, 7},
                {1, 1, 9}
            }
        },
        {
        //dynamic shapes
            {-1, 3, {1, 200}},
            { //target static shapes
                {2, 3, 7},
                {1, 3, 9}
            }
        }
};

/* ============= Convolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0}, {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1} };
std::vector<InputShape> inputShapes2d = {
        {{}, {{ 1, 64, 7, 7 }}},
        {{}, {{ 1, 67, 7, 7 }}},
        {
            //dynamic shape
            { -1, 64, -1, {1, 200} },
            { //target static shapes
                { 2, 64, 7, 7 },
                { 1, 64, 9, 9}
            }
        },
        {
            //dynamic shape
            { -1, 67, -1, {1, 200} },
            { //target static shapes
                { 2, 67, 7, 7 },
                { 1, 67, 9, 9}
            }
        }
};
std::vector<InputShape> inputShapesPlain2Blocked2d = {
        {{}, {{ 1, 1, 7, 7 }}},
        {{}, {{ 1, 2, 7, 7 }}},
        {{}, {{ 1, 3, 7, 7 }}},
        {
            //dynamic shape
            { -1, 1, -1, {1, 200} },
            { //target static shapes
                { 2, 1, 7, 7 },
                { 1, 1, 9, 9}
            }
        },
        {
            //dynamic shape
            { -1, 3, -1, {1, 200} },
            { //target static shapes
                { 2, 3, 7, 7 },
                { 1, 3, 9, 9}
            }
        }
};

/* ============= Convolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0}, {1, 1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<SizeVector> dilations3d = { {1, 1, 1} };
std::vector<InputShape> inputShapes3d = {
        {{}, {{ 1, 64, 7, 7, 7 }}},
        {{}, {{ 1, 67, 7, 7, 7 }}},
        {
            //dynamic shapes
            { -1, 64, -1, {1, 200}, -1 },
            { //target static shapes
                { 1, 64, 7, 7, 7 },
                { 1, 64, 9, 9, 9}
            }
        },
        {
            //dynamic shapes
            { -1, 67, -1, {1, 200}, -1 },
            { //target static shapes
                { 1, 67, 7, 7, 7 },
                { 1, 67, 9, 9, 9}
            }
        }
};
std::vector<InputShape> inputShapesPlain2Blocked3d = {
        {{}, {{ 1, 1, 7, 7, 7 }}},
        {{}, {{ 1, 2, 7, 7, 7 }}},
        {{}, {{ 1, 3, 7, 7, 7 }}},
        {
            //dynamic shapes
            { -1, 1, -1, {1, 200}, -1 },
            { //target static shapes
                { 2, 1, 7, 7, 7 },
                { 1, 1, 9, 9, 9 }
            }
        },
        {
            //dynamic shapes
            { -1, 3, -1, {1, 200}, -1 },
            { //target static shapes
                { 2, 3, 7, 7, 7 },
                { 1, 3, 9, 9, 9 }
            }
        }
};
/* ============= */

/* INSTANCES */
/* ============= Convolution (Gemm 1D) ============= */
const auto convParams_ExplicitPadding_GEMM_1D = ::testing::Combine(
        ::testing::ValuesIn(kernels1d),
        ::testing::ValuesIn(strides1d),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::ValuesIn(dilations1d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_1D = {
        conv_gemm_1D,
        conv_gemm_1D_nspc
};

std::vector<InputShape> inShapesGemm1D = {
        {{}, {{ 2, 12, 7 }}},
        {
            //dynamic shape
            { {1, 200}, 12, {1, 200} },
            { //target static shapes
                { 2, 12, 7 },
                { 1, 12, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_1D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_1D})), // todo: [AV] what about conv_gemm_1D_nspc?
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_1D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (Gemm 2D) ============= */
const auto convParams_ExplicitPadding_GEMM_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_GEMM_2D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_2D = {
        conv_gemm_2D,
        conv_gemm_2D_nspc
};

std::vector<InputShape> inShapesGemm2D = {
        {{}, {{ 2, 12, 7, 7 }}},
        {
            //dynamic shape
            { {1, 200}, 12, -1, {1, 200} },
            { //target static shapes
                { 2, 12, 7, 7 },
                { 1, 12, 5, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (GEMM 3D) ============= */
const auto convParams_ExplicitPadding_GEMM_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_GEMM_3D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::Values(SizeVector{2, 2, 2}),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_3D = {
        conv_gemm_3D,
        conv_gemm_3D_nspc
};

std::vector<InputShape> inShapesGemm3D = {
        {{}, {{ 2, 12, 7, 7, 7 }}},
        {
            //dynamic shape
            { {1, 200}, 12, -1, {1, 200}, -1 },
            { //target static shapes
                { 2, 12, 7, 7, 7 },
                { 1, 12, 5, 5, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (1D) ============= */
const auto convParams_ExplicitPadding_1D = ::testing::Combine(
        ::testing::ValuesIn(kernels1d),
        ::testing::ValuesIn(strides1d),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::ValuesIn(dilations1d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1D = {
        conv_sse42_1D,
        conv_avx2_1D,
        conv_avx512_1D,
        conv_sse42_1D_nspc,
        conv_avx2_1D_nspc,
        conv_avx512_1D_nspc
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_1D})), // todo: [AV] what about conv_avx512_1D_nspc?
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_1D_plain_to_blocked = {
        conv_sse42_plain_to_blocked_1D,
        conv_avx2_plain_to_blocked_1D,
        conv_avx512_plain_to_blocked_1D,
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_1D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (2D) ============= */
const auto convParams_ExplicitPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_2D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_2D = {
        conv_sse42_2D,
        conv_avx2_2D,
        conv_avx512_2D,
        conv_sse42_2D_nspc,
        conv_avx2_2D_nspc,
        conv_avx512_2D_nspc
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D, conv_avx512_2D_nspc})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D, conv_avx512_2D_nspc})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_2D_plain_to_blocked = {
        conv_sse42_plain_to_blocked_2D,
        conv_avx2_plain_to_blocked_2D,
        conv_avx512_plain_to_blocked_2D,
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_2D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_2D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_2D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_2D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Reorder + Convolution ============= */
const auto convParams_Reorder_2D = ::testing::Combine(
        ::testing::Values(SizeVector{1, 1}),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::Values(64),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

std::vector<InputShape> inputShapes_Reorder_2D = {
        {
            // dynamic shape
            { -1, 32, -1, -1 },
            // target static shapes
            {
                { 1, 32, 39, 40 },
                { 2, 32, 20, 20 },
                { 1, 32, 39, 40 },
                { 2, 32, 20, 20 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_reorder_Conv_2D, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Reorder_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes_Reorder_2D),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (3D) ============= */
const auto convParams_ExplicitPadding_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_3D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::Values(SizeVector{2, 2, 2}),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_3D = {
        //conv_sse42_3D, // not supported jit_sse42 for 3d
        conv_avx2_3D,
        conv_avx512_3D,
        conv_avx2_3D_nspc,
        conv_avx512_3D_nspc
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D, conv_avx512_3D_nspc})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D, conv_avx512_3D_nspc})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_3D_plain_to_blocked = {
        conv_avx2_plain_to_blocked_3D,
        conv_avx512_plain_to_blocked_3D,
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_3D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_3D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_3D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_3D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (1D) ============= */

const auto convParams_ExplicitPadding_1x1_1D = ::testing::Combine(
        ::testing::Values(SizeVector({1})),
        ::testing::Values(SizeVector({1})),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(SizeVector({1})),
        ::testing::Values(63),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1x1_1D = {
        conv_sse42_1D_1x1,
        conv_avx2_1D_1x1,
        conv_avx512_1D_1x1,
        conv_sse42_1D_1x1_nspc,
        conv_avx2_1D_1x1_nspc,
        conv_avx512_1D_1x1_nspc
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_1D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_1D_1x1, conv_avx512_2D_1x1_nspc})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_1D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */

const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(63),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1x1_2D = {
        conv_sse42_2D_1x1,
        conv_avx2_2D_1x1,
        conv_avx512_2D_1x1,
        conv_sse42_2D_1x1_nspc,
        conv_avx2_2D_1x1_nspc,
        conv_avx512_2D_1x1_nspc
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1, conv_avx512_2D_1x1_nspc})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Jit Planar ============= */

/* ============= Convolution planar params (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_2D = {
        // sse42 is not supported
        conv_avx2_planar_2D,
        conv_avx512_planar_2D,
};

const auto convParams_Planar_ExplicitPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_Planar_ExplicitPadding_2D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_Jit_Planar_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_2D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_Jit_Planar_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_2D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution planar params (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_3D = {
        // sse42 is not supported
        conv_avx2_planar_3D,
        conv_avx512_planar_3D,
};

const auto convParams_Planar_ExplicitPadding_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::Values(SizeVector{1, 1, 1}),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_Planar_ExplicitPadding_3D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::Values(SizeVector{1, 1, 1}),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::Values(SizeVector{2, 2, 2}),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_Jit_Planar_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_3D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_Jit_Planar_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_3D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution auto padding tests ============= */

const auto convParams_AutoPadding_2D = ::testing::Combine(
        ::testing::Values(kernels2d.front()),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_AutoPad_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_AutoPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= */

} // namespace


/* ============= Winograd ============= */
namespace winograd {

const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        fusingRelu,
        fusingSum,
        fusingAddPerChannel // bias
};

const SizeVector numOutChannels = { 32 };

const std::vector<SizeVector> kernels2d = { {3, 3} };
const std::vector<SizeVector> strides2d = { {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1} };

const auto convParams_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

std::vector<InputShape> inShapesWinograd = {
    {{}, {{ 1, 16, 10, 10 }}},
    {
        //dynamic shape
        { {1, 200}, 16, -1, {1, 200} },
        { //target static shapes
            { 2, 16, 7, 7 },
            { 1, 16, 5, 5 }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_winograd, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesWinograd),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(std::vector<CPUSpecificParams>{conv_winograd})),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

} // namespace winograd

} // namespace CPULayerTestsDefinitions
