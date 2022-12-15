// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include <base/ov_behavior_test_utils.hpp>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"

#include <gtest/gtest.h>

using namespace ov::test::behavior;
namespace {

//
// Executable Network GetMetric
//
class OVClassConfigTestCPU : public ::testing::Test {
public:
    std::shared_ptr<ngraph::Function> model;
    const std::string deviceName = "CPU";

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};

TEST_F(OVClassConfigTestCPU, smoke_GetROPropertiesDoesNotThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (const auto& property : properties) {
        ASSERT_NO_THROW((void)compiledModel.get_property(property));
    }
}

TEST_F(OVClassConfigTestCPU, smoke_SetROPropertiesThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (auto it = properties.begin(); it != properties.end(); ++it) {
        ASSERT_TRUE(it != properties.end());
        ASSERT_FALSE(it->is_mutable());
        ASSERT_THROW(compiledModel.set_property({{*it, "DUMMY VALUE"}}), ov::Exception);
    }
}

TEST_F(OVClassConfigTestCPU, smoke_CheckCoreStreamsHasHigherPriorityThanThroughputHint) {
    ov::Core ie;
    int32_t streams = 1; // throughput hint should apply higher number of streams
    int32_t value;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckCoreStreamsHasHigherPriorityThanLatencyHint) {
    ov::Core ie;
    int32_t streams = 4; // latency hint should apply lower number of streams
    int32_t value;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckModelStreamsHasHigherPriorityThanLatencyHints) {
    ov::Core ie;
    int32_t streams = 4; // latency hint should apply lower number of streams
    int32_t value;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::AnyMap config;
    config[ov::num_streams.name()] = streams;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckModelStreamsHasHigherPriorityThanThroughputHint) {
    ov::Core ie;
    int32_t streams = 1; // throughput hint should apply higher number of streams
    int32_t value;

    ov::AnyMap config;
    config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    config[ov::num_streams.name()] = streams;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

const std::vector<ov::AnyMap> autoModelUnsupportConfigs = {{ov::num_streams(4)},
                                                           {ov::hint::allow_auto_batching(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:CPU"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

const std::vector<DevicePropertiesNumStreamsParams> autoDevicePropertiesConfigsNoThrow = {
    DevicePropertiesNumStreamsParams{"AUTO:CPU", {ov::device::properties("CPU", ov::num_streams(2))}, "CPU"}};

const std::vector<DevicePropertiesNumStreamsParams> autoDevicePropertiesConfigsThrow = {
    DevicePropertiesNumStreamsParams{"AUTO:CPU", {ov::device::properties("CPU", ov::num_streams(2))}, "GPU"}};

const std::vector<DevicePropertiesNumStreamsParams> multiDevicePropertiesConfigsNoThrow = {
    DevicePropertiesNumStreamsParams{"MULTI:CPU", {ov::device::properties("CPU", ov::num_streams(2))}, "CPU"}};
const std::vector<DevicePropertiesNumStreamsParams> multiDevicePropertiesConfigsThrow = {
    DevicePropertiesNumStreamsParams{"MULTI:CPU", {ov::device::properties("CPU", ov::num_streams(2))}, "GPU"}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassAutoExcutableNetowrkGetDevicePropertiesTestNoThrow,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PROPERTIES,
                         ::testing::ValuesIn(multiDevicePropertiesConfigsNoThrow),
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PROPERTIES::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassAutoExcutableNetowrkGetDevicePropertiesTestThrow,
                         OVClassExecutableNetworkGetMetricTestThrow_DEVICE_PROPERTIES,
                         ::testing::ValuesIn(multiDevicePropertiesConfigsThrow),
                         OVClassExecutableNetworkGetMetricTestThrow_DEVICE_PROPERTIES::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_OVClassAutoExcutableNetowrkGetDevicePropertiesTestNoThrow,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PROPERTIES,
                         ::testing::ValuesIn(autoDevicePropertiesConfigsNoThrow),
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PROPERTIES::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_OVClassAutoExcutableNetowrkGetDevicePropertiesTestThrow,
                         OVClassExecutableNetworkGetMetricTestThrow_DEVICE_PROPERTIES,
                         ::testing::ValuesIn(autoDevicePropertiesConfigsThrow),
                         OVClassExecutableNetworkGetMetricTestThrow_DEVICE_PROPERTIES::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_OVClassAutoExcutableNetowrkGetMetricTestThrow,
                         OVClassExecutableNetworkGetMetricTest_UnsupportedConfig,
                         ::testing::Combine(::testing::Values("AUTO:CPU"),
                                            ::testing::ValuesIn(autoModelUnsupportConfigs)),
                         ::testing::PrintToStringParamName());
}  // namespace
