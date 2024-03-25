// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/random_uniform_like.hpp"

#include "exceptions.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {

ov::OutputVector random_uniform_like(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    const auto input = inputs.at(0);

    ov::element::Type target_type;
    if (node.has_attribute("dtype")) {
        const auto dtype = node.get_attribute_value<int64_t>("dtype");
        target_type = common::get_ov_element_type(dtype);
    } else {
        target_type = input.get_element_type();
    }

    const auto target_shape = std::make_shared<v3::ShapeOf>(input);

    const auto high_const = node.get_attribute_as_constant<float>("high", 1.0f);
    const auto low_const = node.get_attribute_as_constant<float>("low", 0.0f);
    const auto seed = node.get_attribute_value<float>("seed", 0.f);

    const uint64_t global_seed = 0;
    const auto seed_uint64 = static_cast<uint64_t>(seed * 1000);

    return {std::make_shared<v8::RandomUniform>(target_shape,
                                                low_const,
                                                high_const,
                                                target_type,
                                                global_seed,
                                                seed_uint64)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
