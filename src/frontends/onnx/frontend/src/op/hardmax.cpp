// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/hardmax.hpp"

#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/topk.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector hardmax(const ov::frontend::onnx::Node& node) {
    const auto input = node.get_ov_inputs().at(0);
    const auto& input_shape = input.get_partial_shape();

    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);
    if (input_shape.rank().is_static()) {
        axis = ov::util::normalize_axis(node.get_description(), axis, input_shape.rank());
    }

    // reshape to 2D - "batch size" x "input feature dimensions" (NxD)
    const auto coerced_tensor = ov::op::util::flatten(input, static_cast<int>(axis));

    const auto coerced_tensor_shape = std::make_shared<ov::op::v0::ShapeOf>(coerced_tensor);
    ov::Output<ov::Node> row_size =
        std::make_shared<v8::Gather>(coerced_tensor_shape,
                                     ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                     ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    row_size = ov::frontend::onnx::reshape::interpret_as_scalar(row_size);

    const auto indices_axis = 1;
    const auto topk = std::make_shared<v11::TopK>(coerced_tensor,
                                                  ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                                  indices_axis,
                                                  ov::op::v11::TopK::Mode::MAX,
                                                  ov::op::v11::TopK::SortType::NONE);

    const auto on_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto off_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    const auto results = std::make_shared<v1::OneHot>(topk->output(1), row_size, on_value, off_value, indices_axis);
    const auto converted_results = std::make_shared<v0::Convert>(results, input.get_element_type());

    const auto output_shape = std::make_shared<ov::op::v0::ShapeOf>(input);
    return {std::make_shared<v1::Reshape>(converted_results, output_shape, false)};
}

}  // namespace set_1
namespace set_13 {
ov::OutputVector hardmax(const ov::frontend::onnx::Node& node) {
    const auto input = node.get_ov_inputs().at(0);
    const auto& input_shape = input.get_partial_shape();

    auto axis = node.get_attribute_value<std::int64_t>("axis", -1);
    axis = ov::util::normalize_axis(node.get_description(), axis, input_shape.rank());

    const auto input_runtime_shape = std::make_shared<ov::op::v0::ShapeOf>(input);
    ov::Output<ov::Node> row_size =
        std::make_shared<v8::Gather>(input_runtime_shape,
                                     ov::op::v0::Constant::create(ov::element::i64, {1}, {axis}),
                                     ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    row_size = ov::frontend::onnx::reshape::interpret_as_scalar(row_size);

    const auto topk = std::make_shared<v11::TopK>(input,
                                                  ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                                  axis,
                                                  ov::op::v11::TopK::Mode::MAX,
                                                  ov::op::v11::TopK::SortType::NONE);

    const auto on_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto off_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    const auto results = std::make_shared<v1::OneHot>(topk->output(1), row_size, on_value, off_value, axis);
    const auto converted_results = std::make_shared<v0::Convert>(results, input.get_element_type());

    const auto output_shape = std::make_shared<ov::op::v0::ShapeOf>(input);
    return {std::make_shared<v1::Reshape>(converted_results, output_shape, false)};
}

}  // namespace set_13
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
