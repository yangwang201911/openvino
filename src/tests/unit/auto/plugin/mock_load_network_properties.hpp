// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>

#include <common_test_utils/test_constants.hpp>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>
#include <iostream>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <openvino/runtime/core.hpp>

#include "ie_icore.hpp"
#include "mock_common.hpp"
#include "plugin.hpp"
#include "plugin/mock_load_network_properties.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"

using namespace MockMultiDevicePlugin;
using ::testing::_;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
namespace MockMultiDevice {

class MockMultiPluginForLoadNetworkWithPropertiesTest : public MultiDeviceInferencePlugin {
public:
    MOCK_METHOD((std::string), GetDeviceList, ((const std::map<std::string, std::string>&)), (const, override));
    MOCK_METHOD(DeviceInformation,
                SelectDevice,
                ((const std::vector<DeviceInformation>&), const std::string&, unsigned int),
                (override));
    MOCK_METHOD((std::list<DeviceInformation>),
                GetValidDevice,
                ((const std::vector<DeviceInformation>&), const std::string&),
                (override));
    MOCK_METHOD((std::vector<DeviceInformation>),
                ParseMetaDevices,
                (const std::string&, (const std::map<std::string, std::string>&)),
                (const, override));
};

using Config = std::map<std::string, std::string>;
class LoadNetworkMockTest {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>> plugin;
    InferenceEngine::CNNNetwork simpleCnnNetwork;
    // mock cpu exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> cpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal> cpuMockExeNetwork;
    NiceMock<MockIInferencePlugin>* cpuMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> cpuMockPlugin;

    // mock gpu exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> gpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal> gpuMockExeNetwork;
    NiceMock<MockIInferencePlugin>* gpuMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> gpuMockPlugin;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>> inferReqInternal;

public:
    void TearDown() {
        core.reset();
        plugin.reset();
    }

    void SetUp() {
        // prepare cpuMockExeNetwork
        cpuMockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        auto cpuMockIPluginPtr = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _))
            .WillByDefault(Return(cpuMockIExeNet));
        cpuMockPlugin = cpuMockIPluginPtr;
        // remove annoying ON CALL message
        EXPECT_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        cpuMockExeNetwork =
            ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(cpuMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

        // prepare gpuMockExeNetwork
        gpuMockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        auto gpuMockIPluginPtr = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _))
            .WillByDefault(Return(gpuMockIExeNet));
        gpuMockPlugin = gpuMockIPluginPtr;
        // remove annoying ON CALL message
        EXPECT_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        gpuMockExeNetwork =
            ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(gpuMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

        // prepare mockicore and cnnNetwork for loading
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        auto* origin_plugin = new NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>();
        plugin = std::shared_ptr<NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>>(origin_plugin);
        // replace core with mock Icore
        plugin->SetCore(core);
        inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        ON_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
        ON_CALL(*gpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));

        ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));
        ON_CALL(*gpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));

        std::vector<std::string> availableDevs = {"CPU", "GPU"};
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));

        std::vector<std::string> metrics = {METRIC_KEY(SUPPORTED_CONFIG_KEYS)};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _)).WillByDefault(Return(metrics));

        ON_CALL(*plugin, ParseMetaDevices)
            .WillByDefault(
                [this](const std::string& priorityDevices, const std::map<std::string, std::string>& config) {
                    return plugin->MultiDeviceInferencePlugin::ParseMetaDevices(priorityDevices, config);
                });

        ON_CALL(*plugin, SelectDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int priority) {
                return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, priority);
            });

        ON_CALL(*plugin, GetValidDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });

        ON_CALL(*plugin, GetDeviceList).WillByDefault([this](const std::map<std::string, std::string>& config) {
            return plugin->MultiDeviceInferencePlugin::GetDeviceList(config);
        });
        ON_CALL(*plugin, SelectDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int Priority) {
                return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, Priority);
            });
        std::vector<std::string> cpuCability{"FP32", "FP16", "INT8", "BIN"};
        std::vector<std::string> gpuCability{"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"};
        ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_CPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(cpuCability));
        ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(gpuCability));

        ON_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(StrEq("CPU")),
                            ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .WillByDefault(Return(cpuMockExeNetwork));

        ON_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(StrEq("GPU")),
                            ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .WillByDefault(Return(gpuMockExeNetwork));

        std::shared_ptr<ngraph::Function> simpleNetwork = ngraph::builder::subgraph::makeSingleConv();
        ASSERT_NO_THROW(simpleCnnNetwork = InferenceEngine::CNNNetwork(simpleNetwork));
    }
};
}  // namespace MockMultiDevice
