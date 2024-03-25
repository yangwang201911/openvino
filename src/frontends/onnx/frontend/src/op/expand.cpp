// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/expand.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector expand(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> data{node.get_ov_inputs().at(0)};
    const ov::Output<ov::Node> shape{node.get_ov_inputs().at(1)};

    if (common::is_failsafe_node(shape.get_node_shared_ptr())) {
        // in case the "shape" input is connected to a failsafe node created in place of an invalid initializer
        // the target shape should be ignored and this Expand operation should not modify its input tensor
        // the Broadcast created below should be eliminated later on by an appropriate optimization pass
        const auto identity_broadcast = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        return {std::make_shared<v3::Broadcast>(data, identity_broadcast, ov::op::BroadcastType::BIDIRECTIONAL)};
    } else {
        return {std::make_shared<v3::Broadcast>(data, shape, ov::op::BroadcastType::BIDIRECTIONAL)};
    }
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
