// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "op/gather_nd.hpp"

#include "openvino/op/gather_nd.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector gather_nd(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector ng_inputs{node.get_ov_inputs()};
    const auto data = ng_inputs.at(0);
    const auto indices = ng_inputs.at(1);
    const auto batch_dims = node.get_attribute_value<int64_t>("batch_dims", 0);

    return {std::make_shared<v8::GatherND>(data, indices, batch_dims)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
