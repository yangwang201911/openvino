// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/infer_request/dynamic_batch.hpp>
#include "common_test_utils/test_constants.hpp"

namespace ConfigurationTestsDefinitions {
namespace {
std::vector<size_t> batch_sizes = {
    16,
    1,
    5,
    9,
    16
};

auto additional_config = []() {
    return std::map<std::string, std::string>{};
};
} // namespace


INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchTest_async, DynamicBatchTest,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(batch_sizes),
        ::testing::Values(true),
        ::testing::Values(additional_config())),
    DynamicBatchTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchTest_sync, DynamicBatchTest,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(batch_sizes),
        ::testing::Values(false),
        ::testing::Values(additional_config())),
    DynamicBatchTest::getTestCaseName);
} // namespace ConfigurationTestsDefinitions
