// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plugin/mock_load_network_properties.hpp"

using ::testing::_;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::Throw;

// define a matcher if all the elements of subMap are contained in the map.
MATCHER_P(MapContains, subMap, "Check if all the elements of the subMap are contained in the map.") {
    if (subMap.empty())
        return true;
    for (auto& item : subMap) {
        auto key = item.first;
        auto value = item.second;
        auto dest = arg.find(key);
        if (dest == arg.end()) {
            return false;
        } else if (dest->second != value) {
            return false;
        }
    }
    return true;
}
using namespace MockMultiDevice;

using ConfigsParam = std::tuple<std::string,               // meta device name to load network
                                std::vector<std::string>,  // hardware device name to expect loading network on
                                Config>;                   // secondary property setting to device

using SecondaryConfigs = ConfigsParam;
static std::vector<SecondaryConfigs> testConfigs;

class LoadNetworkWithSecondaryConfigsMockTest : public ::testing::TestWithParam<SecondaryConfigs>,
                                                public ::MockMultiDevice::LoadNetworkMockTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SecondaryConfigs> obj) {
        std::string deviceName;
        std::vector<std::string> targetDevices;
        Config deviceConfigs;
        std::tie(deviceName, targetDevices, deviceConfigs) = obj.param;
        std::ostringstream result;
        result << "_meta_device_" << deviceName;
        result << "_loadnetwork_to_device_";
        for (auto& device : targetDevices) {
            result << device << "_";
        }
        auto cpuConfig = deviceConfigs.find("CPU");
        auto gpuConfig = deviceConfigs.find("GPU");
        result << "device_properties_";
        if (cpuConfig != deviceConfigs.end())
            result << "CPU_" << cpuConfig->second << "_";
        if (gpuConfig != deviceConfigs.end())
            result << "GPU_" << gpuConfig->second;
        return result.str();
    }

    static std::vector<SecondaryConfigs> CreateSecondaryConfigs() {
        testConfigs.clear();
        testConfigs.push_back(
            SecondaryConfigs{"AUTO", {"CPU"}, {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(SecondaryConfigs{"AUTO",
                                               {"CPU", "GPU"},
                                               {{"GPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        testConfigs.push_back(
            SecondaryConfigs{"AUTO:CPU", {"CPU"}, {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(SecondaryConfigs{"AUTO:CPU,GPU",
                                               {"CPU"},
                                               {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            SecondaryConfigs{"AUTO:GPU", {"GPU"}, {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(SecondaryConfigs{"AUTO:GPU,CPU",
                                               {"CPU", "GPU"},
                                               {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});

        testConfigs.push_back(
            SecondaryConfigs{"MULTI:CPU", {"CPU"}, {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(SecondaryConfigs{"MULTI:CPU,GPU",
                                               {"CPU", "GPU"},
                                               {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            SecondaryConfigs{"MULTI:GPU", {"GPU"}, {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(SecondaryConfigs{"MULTI:GPU,CPU",
                                               {"CPU", "GPU"},
                                               {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        return testConfigs;
    }

    void TearDown() override {
        MockMultiDevice::LoadNetworkMockTest::TearDown();
    }

    void SetUp() override {
        MockMultiDevice::LoadNetworkMockTest::SetUp();
        std::vector<std::string> configKeys = {"SUPPORTED_CONFIG_KEYS",
                                               "NUM_STREAMS",
                                               ov::hint::execution_mode.name(),
                                               ov::hint::performance_mode.name()};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(configKeys));
        ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name()))).WillByDefault(Return(12));
        ON_CALL(*core, GetSupportedConfig)
            .WillByDefault([this](const std::string& device, const std::map<std::string, std::string>& fullConfigs) {
                Config deviceConfigs;
                auto supportedConfigs =
                    core->GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS), {}).as<std::vector<std::string>>();
                for (auto&& item : fullConfigs) {
                    if (item.first.find(device) != std::string::npos) {
                        Config primaryConfigs;
                        std::stringstream strConfigs(item.second);
                        ov::util::Read<Config>{}(strConfigs, primaryConfigs);
                        deviceConfigs.insert(primaryConfigs.begin(), primaryConfigs.end());
                        continue;
                    }
                    if (std::find(supportedConfigs.begin(), supportedConfigs.end(), item.first) !=
                        supportedConfigs.end())
                        deviceConfigs.insert({item.first, item.second});
                }
                return deviceConfigs;
            });
    }
};

using LoadNetworkWithPropertyMockTest = LoadNetworkWithSecondaryConfigsMockTest;
using PrimaryConfigs = ConfigsParam;
const std::vector<SecondaryConfigs> setConfigAndLoadnetwork = {
    SecondaryConfigs{
        "AUTO",
        {"CPU"},
        {{ov::hint::execution_mode.name(), "PERFORMANCE"}, {ov::hint::performance_mode.name(), "THROUGHPUT"}}},
    SecondaryConfigs{"AUTO", {"GPU", "CPU"}, {{ov::hint::execution_mode.name(), "PERFORMANCE"}}},
    SecondaryConfigs{"MULTI:CPU", {"CPU"}, {{ov::hint::execution_mode.name(), "PERFORMANCE"}}},
    SecondaryConfigs{"MULTI:CPU,GPU", {"CPU", "GPU"}, {{ov::hint::execution_mode.name(), "PERFORMANCE"}}}};

TEST_P(LoadNetworkWithPropertyMockTest, LoadNetworkWithPrimaryConfigsCheckTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    Config deviceConfigs;
    std::tie(device, targetDevices, deviceConfigs) = this->GetParam();
    if (device.find("AUTO") != std::string::npos)
        plugin->SetName("AUTO");
    if (device.find("MULTI") != std::string::npos)
        plugin->SetName("MULTI");
    std::ostringstream devicePriorities;
    for (auto&& device : targetDevices) {
        if (device == targetDevices.back())
            devicePriorities << device;
        else
            devicePriorities << device << ",";
    }
    ASSERT_NO_THROW(plugin->SetConfig({{"MULTI_DEVICE_PRIORITIES", devicePriorities.str()}}));
    ASSERT_NO_THROW(plugin->SetConfig(deviceConfigs));

    for (auto& deviceName : targetDevices) {
        EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(deviceName),
                        ::testing::Matcher<const std::map<std::string, std::string>&>(MapContains(deviceConfigs))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, deviceConfigs));
}

TEST_P(LoadNetworkWithSecondaryConfigsMockTest, LoadNetworkWithSecondaryConfigsTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    Config config;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos)
        plugin->SetName("AUTO");
    if (device.find("MULTI") != std::string::npos)
        plugin->SetName("MULTI");

    for (auto& deviceName : targetDevices) {
        auto item = config.find(deviceName);
        Config deviceConfigs;
        if (item != config.end()) {
            std::stringstream strConfigs(item->second);
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<Config>{}(strConfigs, deviceConfigs);
        }
        EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(deviceName),
                        ::testing::Matcher<const std::map<std::string, std::string>&>(MapContains(deviceConfigs))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, config));
}

INSTANTIATE_TEST_SUITE_P(smoke_AutoMock_LoadNetworkWithSecondaryConfigs,
                         LoadNetworkWithSecondaryConfigsMockTest,
                         ::testing::ValuesIn(LoadNetworkWithSecondaryConfigsMockTest::CreateSecondaryConfigs()),
                         LoadNetworkWithSecondaryConfigsMockTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoMock_SetConfigAndLoadNetwork,
                         LoadNetworkWithPropertyMockTest,
                         ::testing::ValuesIn(setConfigAndLoadnetwork),
                         ::testing::PrintToStringParamName());