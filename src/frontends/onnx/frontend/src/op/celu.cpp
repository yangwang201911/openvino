// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/celu.hpp"

#include <memory>

#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/multiply.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector celu(const ov::frontend::onnx::Node& node) {
    auto alpha_node = node.get_attribute_as_constant<float>("alpha", 1.0f);
    auto x_celu = node.get_ov_inputs().at(0);

    auto divide_node = std::make_shared<v1::Divide>(x_celu, alpha_node);
    auto elu_node = std::make_shared<v0::Elu>(divide_node, 1.0);

    return {std::make_shared<v1::Multiply>(alpha_node, elu_node)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
