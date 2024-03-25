// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/depth_to_space.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/depth_to_space.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector depth_to_space(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    const auto& shape = data.get_partial_shape();
    FRONT_END_GENERAL_CHECK(shape.rank().is_static() && shape.rank().get_length() == 4, "Input must be 4-dimensional");

    const auto mode = node.get_attribute_value<std::string>("mode", "DCR");
    v0::DepthToSpace::DepthToSpaceMode ov_mode;
    if (mode == "DCR")
        ov_mode = v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    else if (mode == "CRD")
        ov_mode = v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
    else
        FRONT_END_GENERAL_CHECK(false, "only 'DCR' and 'CRD' modes are supported");

    const auto block_size = node.get_attribute_value<std::int64_t>("blocksize");
    return ov::OutputVector{std::make_shared<v0::DepthToSpace>(data, ov_mode, block_size)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
