// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/tile_broadcast_utils.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class Broadcast : public Node, public TileBroadcastCommon {
public:
    Broadcast(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool created() const override;

    bool neverExecute() const override;
    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needPrepareParams() const override;
    void prepareParams() override;
    bool needShapeInfer() const override;

private:
    void plainExecute(const dnnl::stream& strm);

    enum AutoBroadcastType : uint8_t { NUMPY, EXPLICIT };
    AutoBroadcastType broadcastType = NUMPY;

    static constexpr size_t INPUT_DATA_IDX = 0;
    static constexpr size_t TARGET_SHAPE_IDX = 1;
    static constexpr size_t AXES_MAPPING_IDX = 2;

    std::vector<int32_t> targetShape;
    std::vector<int32_t> axesMapping;
};

}  // namespace ov::intel_cpu::node
