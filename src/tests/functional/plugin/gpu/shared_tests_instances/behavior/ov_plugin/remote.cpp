// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/remote.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {
const std::vector<std::map<std::string, std::string>> configs;


std::vector<std::pair<ov::ParamMap, ov::ParamMap>> generate_remote_params() {
        return {};
}

const std::vector<std::map<std::string, std::string>> MultiConfigs = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> AutoBatchConfigs = {
        {{ CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU}}
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests, OVRemoteTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::element::f32),
                                ::testing::Values(::CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(configs),
                                ::testing::ValuesIn(generate_remote_params())),
                        OVRemoteTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Multi_BehaviorTests, OVRemoteTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::element::f32),
                                ::testing::Values(::CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(MultiConfigs),
                                ::testing::ValuesIn(generate_remote_params())),
                        OVRemoteTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests, OVRemoteTest,
                         ::testing::Combine(
                                 ::testing::Values(ngraph::element::f32),
                                 ::testing::Values(::CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs),
                                 ::testing::ValuesIn(generate_remote_params())),
                         OVRemoteTest::getTestCaseName);
} // namespace
