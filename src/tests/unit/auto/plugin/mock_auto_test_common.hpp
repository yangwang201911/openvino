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

#include "dev/core_impl.hpp"
#include "ie_icore.hpp"
#include "mock_common.hpp"
#include "plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"

using namespace MockMultiDevicePlugin;
using ::testing::_;
using ::testing::MatcherCast;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrEq;
using ::testing::StrNe;
namespace MockMultiDevice {

template <typename T, typename std::enable_if<std::is_base_of<ICore, T>::value, bool>::type = true>
class CoreTestBase {
public:
    typedef T MockCoreType;
    using CorePtr = std::shared_ptr<NiceMock<MockCoreType>>;
    using CoreWeakPtr = std::weak_ptr<NiceMock<MockCoreType>>;
    std::shared_ptr<MultiDeviceInferencePlugin> mockPlugin;
    CorePtr mockCore;

private:
    using NiceMockIPluginPtr = std::shared_ptr<NiceMock<MockIInferencePlugin>>;
    using NiceMockIExeNetPtr = std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>;
    using NiceMockIInferReqPtr = std::shared_ptr<NiceMock<MockIInferRequestInternal>>;
    // HW device mock plugin and mock exeNetwork
    std::map<
        std::string,
        std::tuple<NiceMockIPluginPtr, NiceMockIExeNetPtr, ov::SoPtr<IExecutableNetworkInternal>, NiceMockIInferReqPtr>>
        hwMockObjects;
    std::vector<std::string> availableDevs = {CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU};
    std::map<std::string, std::vector<std::string>> configKeys;
    std::map<std::string, std::vector<std::string>> cability = {
        {CommonTestUtils::DEVICE_CPU, {"FP32", "FP16", "INT8", "BIN"}},
        {CommonTestUtils::DEVICE_GPU, {"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"}}};
    std::vector<std::string> supportedConfigkeys = {"SUPPORTED_CONFIG_KEYS",
                                                    ov::enable_profiling.name(),
                                                    ov::hint::model_priority.name(),
                                                    ov::log::level.name(),
                                                    ov::hint::performance_mode.name(),
                                                    ov::hint::execution_mode.name(),
                                                    ov::hint::num_requests.name(),
                                                    ov::num_streams.name(),
                                                    ov::intel_auto::enable_startup_fallback.name(),
                                                    ov::cache_dir.name(),
                                                    ov::hint::allow_auto_batching.name(),
                                                    ov::auto_batch_timeout.name(),
                                                    ov::device::full_name.name(),
                                                    ov::hint::execution_mode.name(),
                                                    ov::device::priorities.name(),
                                                    ov::device::capabilities.name()};

public:
    std::vector<std::string> getAvailableDevs() {
        return availableDevs;
    }
    std::vector<std::string> getSupportedConfigsKeys() {
        return this->supportedConfigkeys;
    }
    CoreWeakPtr getMockCore() {
        return mockCore;
    }

    void setMockPlugin(const std::weak_ptr<MultiDeviceInferencePlugin> plugin) {
        mockPlugin = plugin.lock();
        ON_CALL(*mockCore, get_plugin(::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_AUTO))))
            .WillByDefault([this](const std::string& pluginName) {
                return ov::Plugin{convert_plugin(mockPlugin), {}};
            });
    }

    void setAvailableDevs(const std::vector<std::string>& availableDevs) {
        this->availableDevs = availableDevs;
        ON_CALL(*mockCore, GetAvailableDevices()).WillByDefault(Return(availableDevs));
        for (auto&& hwDevice : availableDevs) {
            auto hwMockExeNetwork =
                ON_CALL(*mockCore,
                        LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                    ::testing::Matcher<const std::string&>(StrEq(hwDevice)),
                                    ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
                    .WillByDefault(Return(SetUpHWExeNetwork(hwDevice)));
        }
    }
    void setSupportedConfigKeys(const std::string& deviceName, const std::vector<std::string>& configKeys) {
        ON_CALL(*mockCore, GetMetric(StrEq(deviceName), StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
            .WillByDefault(Return(configKeys));
    }
    CoreTestBase() {
        mockCore = std::shared_ptr<NiceMock<MockCoreType>>(new NiceMock<MockCoreType>());

        ON_CALL(*mockCore, GetAvailableDevices()).WillByDefault(Return(availableDevs));

        std::vector<std::string> metrics = {METRIC_KEY(SUPPORTED_CONFIG_KEYS)};
        ON_CALL(*mockCore, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _)).WillByDefault(Return(metrics));

        ON_CALL(*mockCore,
                GetMetric(StrEq(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(cability[CommonTestUtils::DEVICE_GPU]));
        ON_CALL(*mockCore,
                GetMetric(StrNe(CommonTestUtils::DEVICE_CPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(cability[CommonTestUtils::DEVICE_CPU]));

        ON_CALL(*mockCore, GetConfig(_, StrEq(ov::compilation_num_threads.name()))).WillByDefault(Return(12));

        ON_CALL(*mockCore, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
            .WillByDefault(Return(supportedConfigkeys));
        ON_CALL(*mockCore, GetSupportedConfig)
            .WillByDefault([this](const std::string& device, const std::map<std::string, std::string>& fullConfigs) {
                return mockCore->ov::CoreImpl::GetSupportedConfig(device, fullConfigs);
            });
        for (auto&& hwDevice : availableDevs) {
            auto hwMockExeNetwork =
                ON_CALL(*mockCore,
                        LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                    ::testing::Matcher<const std::string&>(StrEq(hwDevice)),
                                    ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
                    .WillByDefault(Return(SetUpHWExeNetwork(hwDevice)));
        }
    }

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> SetUpHWExeNetwork(std::string& hwDevice) {
        // prepare cpuMockExeNetwork
        auto hwMockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        auto hwMockPlugin = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*hwMockPlugin, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(hwMockIExeNet));
        NiceMockIInferReqPtr inferReqInternal;
        inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        ON_CALL(*hwMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
        ON_CALL(*hwMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));
        auto hwIEexecutableNetwork =
            ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(hwMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});
        hwMockObjects[hwDevice] = {hwMockPlugin, hwMockIExeNet, hwIEexecutableNetwork, inferReqInternal};
        return hwIEexecutableNetwork;
    }

    ~CoreTestBase() {
        mockCore.reset();
        hwMockObjects.clear();
    }
};
template <typename T,
          typename std::enable_if<std::is_base_of<MockMultiDeviceInferencePlugin, T>::value, bool>::type = true>
class PluginTestBase {
public:
    typedef T MockPluginType;
    using Ptr = std::shared_ptr<NiceMock<MockPluginType>>;
    using WeakPtr = std::weak_ptr<NiceMock<MockPluginType>>;
    Ptr mockPlugin;

    std::string name;

public:
    void setCore(const std::weak_ptr<ICore> core) {
        mockPlugin->SetCore(core);
    }
    std::string getName() {
        return name;
    }
    void setName(const std::string& deviceName) {
        mockPlugin->SetName(deviceName);
        name = deviceName;
    }

    WeakPtr getMockPlugin() {
        return mockPlugin;
    }

    PluginTestBase() {
        // prepare mockicore and cnnNetwork for loading
        mockPlugin = std::shared_ptr<NiceMock<MockPluginType>>(new NiceMock<MockPluginType>());

        // set the default behavior on the common functions of AUTO/MULTI plugin
        ON_CALL(*mockPlugin, ParseMetaDevices)
            .WillByDefault(
                [this](const std::string& priorityDevices, const std::map<std::string, std::string>& config) {
                    return mockPlugin->MultiDeviceInferencePlugin::ParseMetaDevices(priorityDevices, config);
                });

        ON_CALL(*mockPlugin, GetValidDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });

        ON_CALL(*mockPlugin, GetDeviceList).WillByDefault([this](const std::map<std::string, std::string>& config) {
            return mockPlugin->MultiDeviceInferencePlugin::GetDeviceList(config);
        });
        ON_CALL(*mockPlugin, SelectDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int Priority) {
                return mockPlugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, Priority);
            });
    }

    ~PluginTestBase() {
        mockPlugin.reset();
    }
};  // namespace MockMultiDevice
}  // namespace MockMultiDevice
