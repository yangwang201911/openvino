// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/caching/caching_tests.hpp"
#include <ngraph_ops/nms_ie_internal.hpp>
#include <ngraph_ops/nms_static_shape_ie.hpp>
#include "conformance.hpp"

namespace {
using namespace LayerTestsDefinitions;
using namespace ngraph;

static const std::vector<ov::element::Type> precisionsTemplate = {
        ov::element::f64,
        ov::element::f32,
        ov::element::f16,
        ov::element::i64,
        ov::element::i32,
        ov::element::i16,
        ov::element::i8,
        ov::element::u64,
        ov::element::u32,
        ov::element::u16,
        ov::element::u8,
        ov::element::boolean,
};

static const std::vector<std::size_t> batchSizesTemplate = {
        1, 2
};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_CachingSupportCase, LoadNetworkCacheTestBase,
                         ::testing::Combine(
                                 ::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                 ::testing::ValuesIn(precisionsTemplate),
                                 ::testing::ValuesIn(batchSizesTemplate),
                                 ::testing::Values(ov::test::conformance::targetDevice)),
                         LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
