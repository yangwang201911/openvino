// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/reciprocal.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector reciprocal(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);

    auto one_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {1});
    return {std::make_shared<v1::Divide>(one_node, data)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
