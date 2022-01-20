// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"
#include <cstdint>

namespace BehaviorTestsDefinitions {

std::string DefaultConfigurationTest::getTestCaseName(const ::testing::TestParamInfo<DefaultConfigurationParameters> &obj) {
    std::string targetName;
    DefaultParameter defaultParameter;
    std::tie(targetName, defaultParameter) = obj.param;
    std::ostringstream result;
    result << "configKey=" << defaultParameter._key << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

TEST_P(DefaultConfigurationTest, checkDeviceDefaultConfigurationValue) {
    targetDevice = std::get<DeviceName>(GetParam());
    std::string key;
    InferenceEngine::Parameter parameter;
    CustomComparator customComparator;
    defaultParameter = std::get<DefaultParamterId>(GetParam());
    if (defaultParameter._comparator) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key);
        auto &actual = parameter;
        ASSERT_TRUE(defaultParameter._comparator(expected, actual)) << "For Key: " << defaultParameter._key;
    } else if (defaultParameter._parameter.is<bool>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<bool>();
        auto actual = defaultParameter._parameter.as<bool>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<int>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<int>();
        auto actual = defaultParameter._parameter.as<int>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::uint32_t>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<std::uint32_t>();
        auto actual = defaultParameter._parameter.as<std::uint32_t>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<float>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<float>();
        auto actual = defaultParameter._parameter.as<float>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::string>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<std::string>();
        auto actual = defaultParameter._parameter.as<std::string>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<std::string>>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<std::vector<std::string>>();
        auto actual = defaultParameter._parameter.as<std::vector<std::string>>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<int>>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<std::vector<int>>();
        auto actual = defaultParameter._parameter.as<std::vector<int>>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<std::uint32_t>>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<std::vector<std::uint32_t>>();
        auto actual = defaultParameter._parameter.as<std::vector<std::uint32_t>>();
        ASSERT_EQ(expected, actual);
    } else if (defaultParameter._parameter.is<std::vector<float>>()) {
        auto expected = _core->GetConfig(targetDevice, defaultParameter._key).as<std::vector<float>>();
        auto actual = defaultParameter._parameter.as<std::vector<float>>();
        ASSERT_EQ(expected, actual);
    } else {
        FAIL() << "Unsupported parameter type for key: " << defaultParameter._key;
    }
}



// Setting empty config doesn't throw
TEST_P(EmptyConfigTests, SetEmptyConfig) {
    std::map<std::string, std::string> config;
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
}

TEST_P(EmptyConfigTests, CanLoadNetworkWithEmptyConfig) {
    std::map<std::string, std::string> config;
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, config));
}

TEST_P(CorrectSingleOptionDefaultValueConfigTests, CheckDefaultValueOfConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_EQ(ie->GetConfig(targetDevice, key), value);
}

// Setting correct config doesn't throw
TEST_P(CorrectConfigTests, SetCorrectConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
}

TEST_P(CorrectConfigTests, CanLoadNetworkWithCorrectConfig) {
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

TEST_P(CorrectConfigTests, CanUseCache) {
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    ie->SetConfig({ { CONFIG_KEY(CACHE_DIR), "./test_cache" } });
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    CommonTestUtils::removeDir("./test_cache");
}

TEST_P(CorrectConfigCheck, canSetConfigAndCheckGetConfig) {
    ie->SetConfig(configuration, targetDevice);
    for (const auto& configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = ie->GetConfig(targetDevice, configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(CorrectConfigCheck, canSetConfigTwiceAndCheckGetConfig) {
    ie->SetConfig({}, targetDevice);
    ie->SetConfig(configuration, targetDevice);
    for (const auto& configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = ie->GetConfig(targetDevice, configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(CorrectSingleOptionCustomValueConfigTests, CheckCustomValueOfConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::map<std::string, std::string> configuration = {{key, value}};
    ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
    ASSERT_EQ(ie->GetConfig(targetDevice, key), reference);
}

TEST_P(CorrectConfigPublicOptionsTests, CanSeePublicOption) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    const auto& supportedOptions = metric.as<std::vector<std::string>>();
    ASSERT_NE(std::find(supportedOptions.cbegin(), supportedOptions.cend(), key), supportedOptions.cend());
}

TEST_P(CorrectConfigPrivateOptionsTests, CanNotSeePrivateOption) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    const auto& supportedOptions = metric.as<std::vector<std::string>>();
    ASSERT_EQ(std::find(supportedOptions.cbegin(), supportedOptions.cend(), key), supportedOptions.cend());
}

TEST_P(IncorrectConfigTests, SetConfigWithIncorrectKey) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(ie->SetConfig(configuration, targetDevice), InferenceEngine::Exception);
}

TEST_P(IncorrectConfigTests, CanNotLoadNetworkWithIncorrectConfig) {
    ASSERT_THROW(auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration),
                 InferenceEngine::Exception);
}

TEST_P(IncorrectConfigTests, GetConfigWithIncorrectKey) {
    auto iter = configuration.begin();
    ASSERT_THROW(ie->GetConfig(targetDevice, iter->first), InferenceEngine::Exception);
}

TEST_P(IncorrectConfigSingleOptionTests, CanNotGetConfigWithIncorrectConfig) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(ie->GetConfig(targetDevice, key), InferenceEngine::Exception);
}

TEST_P(IncorrectConfigAPITests, SetConfigWithNoExistingKey) {
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_THROW(ie->SetConfig(configuration, targetDevice), InferenceEngine::Exception);
}


TEST_P(DefaultValuesConfigTests, CanSetDefaultValueBackToPlugin) {
    InferenceEngine::Parameter metric;
    ASSERT_NO_THROW(metric = ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> keys = metric;

    for (auto& key : keys) {
        InferenceEngine::Parameter configValue;
        ASSERT_NO_THROW(configValue = ie->GetConfig(targetDevice, key));

        ASSERT_NO_THROW(ie->SetConfig({{ key, configValue.as<std::string>()}}, targetDevice))
                                    << "device=" << targetDevice << " "
                                    << "config key=" << key << " "
                                    << "value=" << configValue.as<std::string>();
    }
}

} // namespace BehaviorTestsDefinitions
