// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/adaptive_avg_pooling2d.hpp"

#include "exceptions.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector adaptive_avg_pooling2d(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();

    CHECK_VALID_NODE(node, num_inputs == 2, "adaptive_avg_pooling2d expects 2 input tensors. Got: ", num_inputs);

    return {std::make_shared<v8::AdaptiveAvgPool>(inputs[0], inputs[1])};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
