// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/onnx/extension/conversion.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/frontend/onnx/node_context.hpp"

namespace py = pybind11;

using namespace ov::frontend::onnx;

void regclass_frontend_onnx_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> _ext(
        m,
        "_ConversionExtensionONNX",
        py::dynamic_attr());
    class PyConversionExtension : public ConversionExtension {
    public:
        using Ptr = std::shared_ptr<PyConversionExtension>;
        using PyCreatorFunction = std::function<ov::OutputVector(const ov::frontend::NodeContext*)>;
        PyConversionExtension(const std::string& op_type, const PyCreatorFunction& f)
            : ConversionExtension(op_type, [f](const ov::frontend::NodeContext& node) -> ov::OutputVector {
                  return f(static_cast<const ov::frontend::NodeContext*>(&node));
              }) {}
    };
    py::class_<PyConversionExtension, PyConversionExtension::Ptr, ConversionExtension> ext(m,
                                                                                           "ConversionExtensionONNX",
                                                                                           py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const PyConversionExtension::PyCreatorFunction& f) {
        return std::make_shared<PyConversionExtension>(op_type, f);
    }));

    ext.def_property_readonly_static("m_converter", &ConversionExtension::get_converter);
}
