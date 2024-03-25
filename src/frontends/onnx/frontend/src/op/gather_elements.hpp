// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/gather_elements.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
inline ov::OutputVector gather_elements(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ng_inputs{node.get_ov_inputs()};
    auto data = ng_inputs.at(0);
    auto indices = ng_inputs.at(1);
    auto axis = node.get_attribute_value<int64_t>("axis", 0);

    return {std::make_shared<ov::op::v6::GatherElements>(data, indices, axis)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
