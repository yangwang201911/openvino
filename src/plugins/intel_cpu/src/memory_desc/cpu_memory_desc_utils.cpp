// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_ngraph_utils.hpp>
#include "cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

#include <blob_factory.hpp>
#include <cpu_memory.h>
#include <dnnl_types.h>
#include <limits>
#include <numeric>
#include <vector>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

DnnlMemoryDescPtr MemoryDescUtils::convertToDnnlMemoryDesc(const MemoryDescPtr &desc) {
    if (MemoryDescType::Blocked == desc->getType()) {
        const auto cpuDesc = desc->as<CpuBlockedMemoryDesc>();
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(cpuDesc->getPrecision(), cpuDesc->getShape(), cpuDesc->getBlockDims(),
                                                        cpuDesc->getOrder(), cpuDesc->getOffsetPadding(),
                                                        cpuDesc->getOffsetPaddingToData(), cpuDesc->getStrides()));
    } else if (MemoryDescType::Dnnl & desc->getType()) {
        return std::dynamic_pointer_cast<DnnlMemoryDesc>(desc);
    } else {
        OPENVINO_THROW("Cannot convert MemoryDesc to DnnlMemoryDesc");
    }
}

DnnlBlockedMemoryDesc MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const MemoryDesc& desc) {
    if (MemoryDescType::DnnlBlocked == desc.getType()) {
        return DnnlBlockedMemoryDesc(*desc.as<DnnlBlockedMemoryDesc>());
    } else if (MemoryDescType::Blocked == desc.getType()) {
        const auto cpuDesc = desc.as<CpuBlockedMemoryDesc>();
        return DnnlBlockedMemoryDesc(cpuDesc->getPrecision(), cpuDesc->getShape(), cpuDesc->getBlockDims(), cpuDesc->getOrder(), cpuDesc->getOffsetPadding(),
                                     cpuDesc->getOffsetPaddingToData(), cpuDesc->getStrides());
    } else {
        OPENVINO_THROW("Cannot convert MemoryDesc to DnnlMemoryDesc");
    }
}

BlockedMemoryDescPtr MemoryDescUtils::convertToBlockedMemoryDesc(const MemoryDescPtr &desc) {
    if (desc->getType() & MemoryDescType::Blocked) {
        return std::dynamic_pointer_cast<BlockedMemoryDesc>(desc);
    } else {
        OPENVINO_THROW("Can not convert unsupported memory descriptor");
    }
}

CpuBlockedMemoryDescPtr MemoryDescUtils::generateCpuBlockedMemoryDesc(const ov::SoPtr<ov::ITensor>& tensor) {
    const auto& shape = tensor->get_shape().empty() ?  ov::Shape{tensor->get_size()} : tensor->get_shape();

    VectorDims blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);

    auto element_type = tensor->get_element_type();
    const auto& byte_strides = element_type.bitwidth() >= 8 ? tensor->get_strides() : Strides{};

    VectorDims blk_strides;

    if (byte_strides.empty()) {
        blk_strides = ov::row_major_strides(shape);
    } else if (tensor->get_size() == 0) {
        blk_strides.resize(shape.size());
    } else {
        // ROI tensor need figure out correct blk_strides
        blk_strides.resize(byte_strides.size());
        std::transform(byte_strides.begin(),
                       byte_strides.end(),
                       blk_strides.begin(),
                       [&element_type](size_t byte_stride) {
                           OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                           "Limitation: Stride in bytes ",
                                           byte_stride,
                                           " must be divisible by size of element ",
                                           element_type.size());
                           return byte_stride / element_type.size();
                       });
    }

    return std::make_shared<CpuBlockedMemoryDesc>(
        element_type,
        Shape{shape},
        shape,
        blk_order,
        0UL,
        VectorDims{},
        blk_strides);
}

OPENVINO_SUPPRESS_DEPRECATED_START
DnnlBlockedMemoryDesc MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        OPENVINO_THROW("Cannot convert InferenceEngine::TensorDesc with ANY layout to DnnlBlockedMemoryDesc");

    const auto& blkDesc = desc.getBlockingDesc();
    const auto& dims = desc.getDims();

    auto strides = blkDesc.getStrides();
    // for empty tensor case InferenceEngine::TensorDesc fill strides with non zero values before first 0 dims
    // i.e. dims[1, 0, 2, 3] -> strides [0, 6, 3, 1]
    if (std::any_of(dims.begin(), dims.end(), [](size_t dim){ return dim == 0; })) {
        std::fill(strides.begin(), strides.end(), 0);
    }

    return DnnlBlockedMemoryDesc(InferenceEngine::details::convertPrecision(desc.getPrecision()),
                                 Shape(desc.getDims()),
                                 blkDesc.getBlockDims(),
                                 blkDesc.getOrder(),
                                 blkDesc.getOffsetPadding(),
                                 blkDesc.getOffsetPaddingToData(),
                                 strides);
}

InferenceEngine::Blob::Ptr MemoryDescUtils::interpretAsBlob(const IMemory& mem) {
    // TODO [DS]: Rewrite when IE is moved to the new TensorDescriptor
    auto& memDesc = mem.getDesc();
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    desc = InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
    return make_blob_with_precision(desc, mem.getData());
}

InferenceEngine::TensorDesc MemoryDescUtils::interpretAsBlobDesc(const IMemory& mem) {
    auto& memDesc = mem.getDesc();
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    return InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
}

InferenceEngine::TensorDesc MemoryDescUtils::convertToTensorDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        InferenceEngine::BlockingDesc blkDesc =
            desc.getShape().hasZeroDims() ? InferenceEngine::BlockingDesc(blockingDesc->getBlockDims(),
                                                                          blockingDesc->getOrder(),
                                                                          blockingDesc->getOffsetPadding(),
                                                                          blockingDesc->getOffsetPaddingToData())
                                          : InferenceEngine::BlockingDesc(blockingDesc->getBlockDims(),
                                                                          blockingDesc->getOrder(),
                                                                          blockingDesc->getOffsetPadding(),
                                                                          blockingDesc->getOffsetPaddingToData(),
                                                                          blockingDesc->getStrides());
        return InferenceEngine::TensorDesc(InferenceEngine::details::convertPrecision(blockingDesc->getPrecision()),
                                           blockingDesc->getShape().getStaticDims(),
                                           blkDesc);
    } else {
        OPENVINO_THROW("Cannot convert MemoryDesc to InferenceEngine::TensorDesc");
    }
}
OPENVINO_SUPPRESS_DEPRECATED_END

std::string MemoryDescUtils::dim2str(Dim dim) {
    return dim == Shape::UNDEFINED_DIM ? "?" : std::to_string(dim);
}

std::string MemoryDescUtils::dims2str(const VectorDims& dims) {
    std::stringstream output;
    output << "{";

    if (!dims.empty()) {
        auto itr = dims.begin();
        do {
            output << dim2str(*itr);
        } while (++itr != dims.end() && output << ", ");
    }

    output << "}";
    return output.str();
}

std::shared_ptr<MemoryDesc> MemoryDescUtils::makeDummyDesc(const MemoryDesc &desc, Dim dummyVal) {
    auto dummyShape = makeDummyShape(desc.getShape(), dummyVal);
    return desc.cloneWithNewDims(dummyShape.getStaticDims());
}

Shape MemoryDescUtils::makeDummyShape(const Shape &shape, Dim dummyVal) {
    const auto& minDims = shape.getMinDims();
    const auto& maxDims = shape.getMaxDims();
    const auto& dims = shape.getDims();
    VectorDims dummyDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dummyDims[i] = dims[i] == Shape::UNDEFINED_DIM ? std::min(maxDims[i], std::max(minDims[i], dummyVal)) : dims[i];
    }
    return Shape(dummyDims);
}

Shape MemoryDescUtils::makeDummyShape(const Shape &shape, const VectorDims& dummyVals) {
    if (shape.getRank() != dummyVals.size()) {
        OPENVINO_THROW("makeDummyShape(): dummyVals vector size and shape ranks mismatch");
    }
    const auto& minDims = shape.getMinDims();
    const auto& maxDims = shape.getMaxDims();
    const auto& dims = shape.getDims();
    VectorDims dummyDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dummyDims[i] = dims[i] == Shape::UNDEFINED_DIM ? std::min(maxDims[i], std::max(minDims[i], dummyVals[i])) : dims[i];
    }
    return Shape(dummyDims);
}
}   // namespace intel_cpu
}   // namespace ov
