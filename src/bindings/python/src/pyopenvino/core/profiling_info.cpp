// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/profiling_info.hpp"

#include <pybind11/chrono.h>

#include "openvino/runtime/profiling_info.hpp"

namespace py = pybind11;

void regclass_ProfilingInfo(py::module m) {
    py::class_<ov::ProfilingInfo, std::shared_ptr<ov::ProfilingInfo>> cls(m, "ProfilingInfo");
    cls.def(py::init<>())
        .def_readwrite("status", &ov::ProfilingInfo::status)
        .def_readwrite("real_time", &ov::ProfilingInfo::real_time)
        .def_readwrite("cpu_time", &ov::ProfilingInfo::cpu_time)
        .def_readwrite("node_name", &ov::ProfilingInfo::node_name)
        .def_readwrite("exec_type", &ov::ProfilingInfo::exec_type)
        .def_readwrite("node_type", &ov::ProfilingInfo::node_type);

    py::enum_<ov::ProfilingInfo::Status>(cls, "Status")
        .value("NOT_RUN", ov::ProfilingInfo::Status::NOT_RUN)
        .value("OPTIMIZED_OUT", ov::ProfilingInfo::Status::OPTIMIZED_OUT)
        .value("EXECUTED", ov::ProfilingInfo::Status::EXECUTED)
        .export_values();
}
