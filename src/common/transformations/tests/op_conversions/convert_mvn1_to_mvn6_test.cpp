// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_mvn1_to_mvn6.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6) {
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, false, true, 1e-5);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 3});
        auto mvn =
            std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6_across_channels) {
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, true, true, 1e-5);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 2, 3});
        auto mvn =
            std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6_5D) {
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4, 5});
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, false, true, 1e-5);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4, 5});
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {2, 3, 4});
        auto mvn =
            std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
}
