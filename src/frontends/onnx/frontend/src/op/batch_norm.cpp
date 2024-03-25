// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/batch_norm.hpp"

#include <cstdint>
#include <memory>

#include "core/null_node.hpp"
#include "exceptions.hpp"
#include "openvino/op/batch_norm.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
// This version supports ONNX BatchNormalization-1 and BatchNormalization-6
ov::OutputVector batch_norm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);
    ov::Output<ov::Node> mean;
    ov::Output<ov::Node> var;

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};

    // Currently only BatchNormalization inference mode is supported by OpenVINO
    std::int64_t is_test{node.get_attribute_value<std::int64_t>("is_test", 1)};
    CHECK_VALID_NODE(node, is_test, "only 'is_test' mode is supported.");

    // optional outputs
    auto after_bn_mean = std::make_shared<NullNode>();
    auto after_bn_var = std::make_shared<NullNode>();
    auto saved_mean = std::make_shared<NullNode>();
    auto saved_var = std::make_shared<NullNode>();

    if (inputs.size() >= 5) {
        mean = inputs.at(3);
        var = inputs.at(4);
        return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon),
                after_bn_mean,
                after_bn_var,
                saved_mean,
                saved_var};
    }

    OPENVINO_THROW("Cannot create OpenVINO batch norm with unsupported number of inputs");
}
}  // namespace set_1

namespace set_7 {
// This version supports ONNX BatchNormalization-7 and BatchNormalization-9
ov::OutputVector batch_norm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto scale = inputs.at(1);
    auto bias = inputs.at(2);
    auto mean = inputs.at(3);
    auto var = inputs.at(4);

    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
    // Attribute "spatial" is ignored, as we only support inference mode of
    // BatchNormalization

    CHECK_VALID_NODE(node, node.get_outputs_size() == 1, "Training mode of BatchNormalization is not supported.");

    return {std::make_shared<v5::BatchNormInference>(x, scale, bias, mean, var, epsilon)};
}

}  // namespace set_7
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
