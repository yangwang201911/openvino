// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/experimental_detectron/roi_feature_extractor.hpp"

#include "core/node.hpp"
#include "openvino/op/experimental_detectron_roi_feature.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector experimental_detectron_roi_feature_extractor(const ov::frontend::onnx::Node& node) {
    using ROIFeatureExtractor = v6::ExperimentalDetectronROIFeatureExtractor;

    auto inputs = node.get_ov_inputs();

    ROIFeatureExtractor::Attributes attrs{};
    attrs.output_size = node.get_attribute_value<std::int64_t>("output_size", 7);
    attrs.sampling_ratio = node.get_attribute_value<std::int64_t>("sampling_ratio", 2);
    attrs.aligned = static_cast<bool>(node.get_attribute_value<std::int64_t>("aligned", 0));
    attrs.pyramid_scales = node.get_attribute_value<std::vector<std::int64_t>>("pyramid_scales", {4, 8, 16, 32, 64});
    auto roi_feature_extractor = std::make_shared<ROIFeatureExtractor>(inputs, attrs);
    return {roi_feature_extractor->output(0), roi_feature_extractor->output(1)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
