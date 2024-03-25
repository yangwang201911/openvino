// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/util/common_util.hpp"

TEST(get_constant_from_source, invalidation_check) {
    auto a = ov::opset8::Constant::create(ov::element::i64, {100}, {123});
    auto b = ov::opset8::Constant::create(ov::element::i64, {1}, {123});
    auto div = std::make_shared<ov::opset8::Divide>(a, b);
    auto s = std::make_shared<ov::opset8::ShapeOf>(a);
    auto r = std::make_shared<ov::opset8::Reshape>(div, s, true);
    auto tmp_consumer = std::make_shared<ov::opset8::ShapeOf>(s);

    ASSERT_TRUE(ov::util::get_constant_from_source(r));

    ASSERT_TRUE(r->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(r->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(s->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(s->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(b->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(b->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(a->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(a->get_output_tensor(0).get_upper_value());

    ASSERT_FALSE(div->get_output_tensor(0).get_lower_value());
    ASSERT_FALSE(div->get_output_tensor(0).get_upper_value());
}

TEST(get_constant_from_source, extract_static_dim_from_dynamic_shape_check) {
    auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 128});
    auto shape = std::make_shared<ov::opset8::ShapeOf>(data);
    auto one = ov::opset8::Constant::create(ov::element::i64, {1}, {1});
    auto zero = ov::opset8::Constant::create(ov::element::i64, {1}, {0});
    const auto extract_static_dimension = std::make_shared<ov::opset8::Gather>(shape, one, zero);

    ASSERT_TRUE(ov::util::get_constant_from_source(extract_static_dimension));

    ASSERT_TRUE(extract_static_dimension->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(extract_static_dimension->get_output_tensor(0).get_upper_value());
}

TEST(constantfold_subgraph, split) {
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto constant = ov::opset8::Constant::create(ov::element::f32, ov::Shape{input.size()}, input);
    auto mul = std::make_shared<ov::opset8::Multiply>(constant,
                                                      ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {1}));
    auto shape = std::make_shared<ov::opset8::ShapeOf>(mul);
    auto len_0 =
        std::make_shared<ov::opset8::Divide>(shape, ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    auto len_1 = std::make_shared<ov::opset8::Subtract>(shape, len_0);
    auto lenghts = std::make_shared<ov::opset8::Concat>(ov::OutputVector{len_0, len_1}, 0);
    auto axis = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto split = std::make_shared<ov::opset8::VariadicSplit>(mul, axis, lenghts);
    std::vector<float> expected(std::next(input.begin(), input.size() / 2), input.end());
    auto ret = ov::util::constantfold_subgraph(split->output(1));
    ASSERT_NE(ret, nullptr);
    auto actual = ret->cast_vector<float>();
    ASSERT_EQ(expected, actual);
}

TEST(constantfold_subgraph, shapeof) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, -1});
    auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(param);
    auto zero = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto two = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2});
    auto stop = std::make_shared<ov::op::v8::Slice>(shapeof, one /*start*/, two /*stop*/, one /*step*/, zero /*axis*/);
    auto slice = std::make_shared<ov::op::v8::Slice>(param, one /*start*/, stop, one /*step*/, one /*axis*/);

    auto ret = ov::util::constantfold_subgraph(stop);
    ASSERT_NE(ret, nullptr);
    auto actual = ret->cast_vector<int64_t>();
    std::vector<int64_t> expected{3};
    ASSERT_EQ(expected, actual);
}
