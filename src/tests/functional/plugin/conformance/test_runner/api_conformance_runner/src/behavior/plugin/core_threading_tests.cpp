// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/plugin/core_threading.hpp>
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;

const Params coreThreadingParams[] = {
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_HETERO, generateConfigs(CommonTestUtils::DEVICE_HETERO).front() },
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_MULTI, generateConfigs(CommonTestUtils::DEVICE_MULTI).front() },
    std::tuple<Device, Config>{ CommonTestUtils::DEVICE_AUTO, generateConfigs(CommonTestUtils::DEVICE_AUTO).front() },
};

INSTANTIATE_TEST_SUITE_P(Conformance_, CoreThreadingTests, testing::ValuesIn(coreThreadingParams), CoreThreadingTests::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(Conformance, CoreThreadingTests,
        ::testing::Combine(
                ::testing::Values(ov::test::conformance::targetDevice),
                ::testing::Values(Config{{ CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) }})),
        CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conformance, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(coreThreadingParams),
                     testing::Values(4),
                     testing::Values(50),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);

}  // namespace
