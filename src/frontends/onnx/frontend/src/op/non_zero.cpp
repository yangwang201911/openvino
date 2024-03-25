// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/non_zero.hpp"

#include "openvino/op/non_zero.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector non_zero(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v3::NonZero>(data, ov::element::i64)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
