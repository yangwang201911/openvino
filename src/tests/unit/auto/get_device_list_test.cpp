// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common_test_utils/test_constants.hpp>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

#include "cpp/ie_plugin.hpp"
#include "mock_common.hpp"
#include "plugin/mock_auto_device_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"

using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;
Config config = {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ""}};

class FunctionsLinesCoverageTest : public ::testing::TestWithParam<Config> {
public:
    std::shared_ptr<MockICore> core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> mockPlugin;
    std::shared_ptr<MultiDeviceInferencePlugin> plugin;
    std::shared_ptr<ngraph::Function> actualNetwork;
    InferenceEngine::CNNNetwork actualCnnNetwork;
    InferenceEngine::Core ie;

public:
    static std::string getTestCaseName(testing::TestParamInfo<Config> obj) {
        std::ostringstream result;
        result << "AutoFunctionsLinesCoverageTest";
        return result.str();
    }

    void TearDown() override {
        core.reset();
        mockPlugin.reset();
        plugin.reset();
        mockPlugin.reset();
    }

    void SetUp() override {
        // prepare mockicore and cnnNetwork for loading
        core = std::shared_ptr<MockICore>(new MockICore());
        auto* origin_plugin = new MockMultiDeviceInferencePlugin();
        mockPlugin = std::shared_ptr<MockMultiDeviceInferencePlugin>(origin_plugin);
        // Create Generic network
        actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
        ASSERT_NO_THROW(actualCnnNetwork = InferenceEngine::CNNNetwork(actualNetwork));
    }
};

TEST_P(FunctionsLinesCoverageTest, GetDevicesListWithInvaildConfigThrow) {
    // get Parameter
    auto config = this->GetParam();
    ASSERT_THROW(mockPlugin->GetDeviceList(config), InferenceEngine::Exception);
}

TEST_P(FunctionsLinesCoverageTest, FilterDeviceWithEmptyMetaDevicesListThrow) {
    // Set cpnfig
    Config invaildConfig = {{CONFIG_KEY_INTERNAL(MULTI_WORK_MODE_AS_AUTO), "YES"}};
    // get Parameter
    const std::vector<DeviceInformation> metaDevices;
    ASSERT_THROW(ie.LoadNetwork(actualCnnNetwork), InferenceEngine::Exception);
}

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, FunctionsLinesCoverageTest,
                         ::testing::Values(config),
                         FunctionsLinesCoverageTest::getTestCaseName);