// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels.h"

#include <cmath>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <openvino/op/shuffle_channels.hpp>
#include <set>
#include <string>

#include "common/primitive_hashing_utils.hpp"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/permute_kernel.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::node {

size_t ShuffleChannels::ShuffleChannelsAttributes::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, layoutType);
    seed = hash_combine(seed, dataRank);
    seed = hash_combine(seed, axis);
    seed = hash_combine(seed, spatialRank);
    seed = hash_combine(seed, group);
    seed = hash_combine(seed, dataSize);
    seed = get_vector_hash(seed, srcDims);
    seed = get_vector_hash(seed, srcBlockedDims);

    return seed;
}

bool ShuffleChannels::ShuffleChannelsAttributes::operator==(const ShuffleChannelsAttributes& rhs) const {
    bool result = layoutType == rhs.layoutType && dataRank == rhs.dataRank && axis == rhs.axis &&
                  spatialRank == rhs.spatialRank && group == rhs.group && dataSize == rhs.dataSize &&
                  srcDims == rhs.srcDims && srcBlockedDims == rhs.srcBlockedDims;
    return result;
}

bool ShuffleChannels::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        auto shuffleChannels = ov::as_type_ptr<const ov::op::v0::ShuffleChannels>(op);
        if (!shuffleChannels) {
            errorMessage = "Only opset1 ShuffleChannels operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ShuffleChannels::ShuffleChannels(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (inputShapes.size() != 1 || outputShapes.size() != 1) {
        CPU_NODE_THROW("has incorrect number of input/output edges.");
    }

    auto shuffleChannels = ov::as_type_ptr<const ov::op::v0::ShuffleChannels>(op);
    attrs.group = shuffleChannels->get_group();
    attrs.axis = static_cast<int>(shuffleChannels->get_axis());
    attrs.dataRank = getInputShapeAtPort(0).getRank();
    if (attrs.axis < 0) {
        attrs.axis += attrs.dataRank;
    }
}

void ShuffleChannels::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);
    const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8, 16};
    if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end()) {
        CPU_NODE_THROW("has unsupported precision: ", precision.get_type_name());
    }

    auto impl_type = []() {
        if (mayiuse(cpu::x64::avx512_core)) {
            return impl_desc_type::jit_avx512;
        }
        if (mayiuse(cpu::x64::avx2)) {
            return impl_desc_type::jit_avx2;
        }
        if (mayiuse(cpu::x64::sse41)) {
            return impl_desc_type::jit_sse42;
        }
        return impl_desc_type::ref;
    }();

    // use ncsp as default for non-quantized networks and nspc for quantized
    auto firstCreatorType = context->isGraphQuantized() ? LayoutType::nspc : LayoutType::ncsp;
    auto secondCreatorType = context->isGraphQuantized() ? LayoutType::ncsp : LayoutType::nspc;

    addSupportedPrimDesc({{firstCreatorType, precision}}, {{firstCreatorType, precision}}, impl_type);
    addSupportedPrimDesc({{secondCreatorType, precision}}, {{secondCreatorType, precision}}, impl_type);
    // canUseBlocked
    if (attrs.axis != 1) {
        addSupportedPrimDesc({{LayoutType::nCsp8c, precision}}, {{LayoutType::nCsp8c, precision}}, impl_type);
        addSupportedPrimDesc({{LayoutType::nCsp16c, precision}}, {{LayoutType::nCsp16c, precision}}, impl_type);
    }
}

void ShuffleChannels::createPrimitive() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);
    CPU_NODE_ASSERT(dstMemPtr, "has null destination memory");
    CPU_NODE_ASSERT(srcMemPtr, "has null input memory");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor(), "has unidentified preferable primitive descriptor");

    const auto& memoryDesc = srcMemPtr->getDesc();
    attrs.spatialRank = attrs.dataRank - attrs.axis - 1;
    attrs.dataSize = memoryDesc.getPrecision().size();
    if (memoryDesc.hasLayoutType(LayoutType::nCsp16c)) {
        attrs.layoutType = LayoutType::nCsp16c;
    } else if (memoryDesc.hasLayoutType(LayoutType::nCsp8c)) {
        attrs.layoutType = LayoutType::nCsp8c;
    } else if (memoryDesc.hasLayoutType(LayoutType::nspc)) {
        attrs.layoutType = LayoutType::nspc;
    } else {
        attrs.layoutType = LayoutType::ncsp;
    }

    if (inputShapesDefined() && isExecutable()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void ShuffleChannels::prepareParams() {
    auto srcMemPtr = getSrcMemoryAtPort(0);
    auto builder = [](const ShuffleChannelsAttributes& key) -> std::shared_ptr<ShuffleChannelsExecutor> {
        return std::make_shared<ShuffleChannelsExecutor>(key);
    };
    attrs.srcDims = srcMemPtr->getStaticDims();
    attrs.srcBlockedDims = srcMemPtr->getDescWithType<BlockedMemoryDesc>()->getBlockDims();

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(attrs, builder);
    CPU_NODE_ASSERT(result.first, "executor was not found for node.");

    execPtr = result.first;
}

ShuffleChannels::ShuffleChannelsExecutor::ShuffleChannelsExecutor(const ShuffleChannelsAttributes& attrs) {
    OPENVINO_ASSERT(
        one_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c, LayoutType::nspc, LayoutType::ncsp),
        "ShuffleChannels executor supports only 'nCsp16c', 'nCsp8c', 'nspc' or 'ncsp' layouts.");

    const bool isBlocked = one_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c);
    const bool isChannelsLast = attrs.layoutType == LayoutType::nspc;
    const auto& srcDims = attrs.srcDims;
    const auto& srcBlockedDims = attrs.srcBlockedDims;

    // 2 for decomposed axis dim, 1 for composed spatial dim
    const int batchRank = attrs.axis;
    const int reshapedRank = batchRank + 2 + static_cast<int>(attrs.spatialRank != 0) +
                             static_cast<int>(isBlocked && (attrs.spatialRank == 0));
    PermuteParams params;
    params.data_size = attrs.dataSize;
    params.order.resize(reshapedRank, 0);
    params.src_block_order.resize(reshapedRank);
    params.dst_block_order.resize(reshapedRank);
    params.dst_block_dims.resize(reshapedRank);
    params.src_block_dims.resize(reshapedRank);

    const size_t groupSize = srcDims[attrs.axis] / attrs.group;
    size_t spatialShapeSize = 1;
    if (attrs.spatialRank != 0) {
        for (int i = batchRank + 1; i < attrs.dataRank; i++) {
            spatialShapeSize *= srcDims[i];
        }
    }

    auto decomposeAndTranpose = [&](int axis) {
        params.src_block_dims[axis] = attrs.group;
        params.src_block_dims[axis + 1] = groupSize;
        params.order[axis] = axis + 1;
        params.order[axis + 1] = axis;
    };

    const int channelDim = 1;
    if (isBlocked) {
        size_t blkSize = srcBlockedDims.back();
        size_t CB = srcBlockedDims[1];
        if (attrs.axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                params.order[i] = i;
                params.src_block_dims[i] = srcBlockedDims[i];
            }
            decomposeAndTranpose(batchRank);

            params.order[batchRank + 2] = batchRank + 2;
            params.src_block_dims[batchRank + 2] = spatialShapeSize * blkSize;
        } else {  // axis on batch
            decomposeAndTranpose(0);
            spatialShapeSize = CB * blkSize;
            for (int i = 2; i < attrs.dataRank; i++) {
                spatialShapeSize *= srcDims[i];
            }
            params.order[2] = 2;
            params.src_block_dims[2] = spatialShapeSize;
        }
    } else if (isChannelsLast) {
        if (attrs.axis == channelDim) {  // axis on channel
            params.order[0] = 0;
            params.src_block_dims[0] = srcDims[0];
            params.order[1] = 1;
            params.src_block_dims[1] = spatialShapeSize;
            decomposeAndTranpose(2);
        } else if (attrs.axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                if (i == 0) {
                    params.order[i] = i;
                    params.src_block_dims[i] = srcDims[i];
                } else if (i == 1) {
                    params.order[reshapedRank - 1] = reshapedRank - 1;
                    params.src_block_dims[params.order[reshapedRank - 1]] = srcDims[i];
                } else if (i > 1) {
                    params.order[i - 1] = i - 1;
                    params.src_block_dims[i - 1] = srcDims[i];
                }
            }
            decomposeAndTranpose(batchRank - 1);

            if (attrs.spatialRank != 0) {
                params.order[batchRank + 1] = batchRank + 1;
                params.src_block_dims[batchRank + 1] = spatialShapeSize;
            }
        } else {  // axis on batch
            decomposeAndTranpose(0);
            params.order[2] = 2;
            params.src_block_dims[2] = spatialShapeSize;
        }
    } else {
        for (int i = 0; i < batchRank; i++) {
            params.src_block_dims[i] = srcDims[i];
            params.order[i] = i;
        }

        decomposeAndTranpose(batchRank);
        if (attrs.spatialRank != 0) {
            params.order[batchRank + 2] = batchRank + 2;
            params.src_block_dims[batchRank + 2] = spatialShapeSize;
        }
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (int i = 0; i < reshapedRank; i++) {
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];
    }

    permuteKernel = std::make_unique<PermuteKernel>(params);
}

void ShuffleChannels::ShuffleChannelsExecutor::exec(const uint8_t* srcData, uint8_t* dstData, int MB) {
    OPENVINO_ASSERT(permuteKernel, "Could not execute. Kernel for Transpose node was not compiled.");

    if (MB > 0) {
        permuteKernel->execute(srcData, dstData, MB);
    } else {
        permuteKernel->execute(srcData, dstData);
    }
}

void ShuffleChannels::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void ShuffleChannels::execute([[maybe_unused]] const dnnl::stream& strm) {
    CPU_NODE_ASSERT(execPtr, "doesn't have a compiled executor.");

    int MB = (attrs.axis != 0) ? getSrcMemoryAtPort(0)->getStaticDims()[0] : -1;

    const auto* srcData = getSrcDataAtPortAs<const uint8_t>(0);
    auto* dstData = getDstDataAtPortAs<uint8_t>(0);
    execPtr->exec(srcData, dstData, MB);
}

bool ShuffleChannels::created() const {
    return getType() == Type::ShuffleChannels;
}

}  // namespace ov::intel_cpu::node
