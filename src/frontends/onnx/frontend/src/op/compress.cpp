// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/compress.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector compress(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto condition = node.get_ov_inputs().at(1);

    int64_t axis = 0;
    if (node.has_attribute("axis")) {
        axis = node.get_attribute_value<int64_t>("axis");
    } else {
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
        data = std::make_shared<v0::Squeeze>(ov::op::util::flatten(data, static_cast<int>(axis)));
    }
    auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    auto zero_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto result =
        std::make_shared<v8::Gather>(data,
                                     std::make_shared<v0::Squeeze>(std::make_shared<v3::NonZero>(condition), zero_node),
                                     axis_node);

    return {result};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
