// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
/// \brief      Creates OV node representing ONNX LpNormalization operator.
///
///             Suppose A contains spatial dimensions of input tensor, then
///             for matrix A we have p-norm defined as following double sum over
///             all elements:
///             ||A||_p = ||vec(A)||_p =
///                  [sum_{i=1}^m sum_{j=1}^n abs(a_{i,j})^p]^{1/p}
///
/// \param[in]  node  The input ONNX node representing this operation.
///
/// \return     Vector of nodes containting resulting OV nodes.
///
ov::OutputVector lp_norm(const ov::frontend::onnx::Node& node);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
