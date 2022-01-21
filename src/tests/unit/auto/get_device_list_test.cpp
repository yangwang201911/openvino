// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "plugin/mock_auto_device_plugin.hpp"
#include "cpp/ie_plugin.hpp"
#include "mock_common.hpp"

using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

TEST(GetDeviceListTest, ReturnEmptyDeviceListWithThrow) {
    auto* origin_plugin = new MockMultiDeviceInferencePlugin();
    Config config;
    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ""});
    ASSERT_THROW(origin_plugin->GetDeviceList(config), InferenceEngine::Exception);
}