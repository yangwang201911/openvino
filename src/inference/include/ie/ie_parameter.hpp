// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Parameter class
 * @file ie_parameter.hpp
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#include "ie_blob.h"
#include "openvino/runtime/parameter.hpp"

namespace InferenceEngine {

/**
 * @brief Alias for type that can store any value
 */
using Parameter = ov::Any;
using ov::ParamMap;

}  // namespace InferenceEngine
