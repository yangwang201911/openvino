// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "cpu_types.h"
#include "ctc_loss.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

bool CTCLoss::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto ctcLossOp = ov::as_type_ptr<const ov::op::v4::CTCLoss>(op);
        if (!ctcLossOp) {
            errorMessage = "Node is not an instance of the CTCLoss operation from operation set v4.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

CTCLoss::CTCLoss(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (getOriginalInputsNumber() != 4 && getOriginalInputsNumber() != 5) {
        CPU_NODE_THROW("has invalid inputs number.");
    }

    auto ctcLossOp = ov::as_type_ptr<const ov::op::v4::CTCLoss>(op);
    ctcMergeRepeated = ctcLossOp->get_ctc_merge_repeated();
    preprocessCollapseRepeated = ctcLossOp->get_preprocess_collapse_repeated();
    unique = ctcLossOp->get_unique();
}

void CTCLoss::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    inDataConf.emplace_back(LayoutType::ncsp, ov::element::f32);
    for (size_t i = 1; i < inputShapes.size(); ++i) {
        inDataConf.emplace_back(LayoutType::ncsp, ov::element::i32);
    }

    addSupportedPrimDesc(inDataConf, {{LayoutType::ncsp, ov::element::f32}}, impl_desc_type::ref_any);
}

void CTCLoss::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void CTCLoss::execute([[maybe_unused]] const dnnl::stream& strm) {
    int32_t returnCode = 0;

    const auto* logits = getSrcDataAtPortAs<const float>(0);
    const auto* logitsLength = getSrcDataAtPortAs<const int>(1);
    const auto* labels = getSrcDataAtPortAs<const int>(2);
    const auto* labelsLength = getSrcDataAtPortAs<const int>(3);
    auto* dstData = getDstDataAtPortAs<float>(0);

    const auto& inDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    const size_t batchNum = inDims[0];
    const size_t maxTime = inDims[1];
    const size_t classesNum = inDims[2];

    int blankIndex = classesNum - 1;
    if (inputShapes.size() > 4) {
        blankIndex = getSrcDataAtPortAs<const int>(4)[0];
    }

    std::vector<int> decodedTargetLenB(batchNum, 0);
    std::vector<std::vector<int>> targetDB(batchNum);
    std::vector<std::vector<std::vector<float>>> logProbabilitiesB(batchNum);
    const auto threads_num = parallel_get_max_threads();
    std::vector<std::string> errorMsgB(threads_num);

    auto threadBody_1 = [&](const int ithr, const int nthr) {
        size_t start(0LU);
        size_t end(0LU);
        splitter(batchNum, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }

        for (size_t b = start; b < end; b++) {
            if (logitsLength[b] < 0 || labelsLength[b] < 0 || logitsLength[b] > static_cast<int>(maxTime) ||
                labelsLength[b] > logitsLength[b]) {
                errorMsgB[ithr] = std::string("Logit length cannot be greater than max sequence length. ") +
                                  "Label length cannot be greater than a logit length" +
                                  " and both cannot be negative.\nMaxSeqLen: " + std::to_string(maxTime) +
                                  "; Logit len: " + std::to_string(logitsLength[b]) +
                                  "; Label len: " + std::to_string(labelsLength[b]);
                returnCode = -1;
                return;
            }
            const size_t actualLogitLen = logitsLength[b];
            const size_t actualTargetLen = labelsLength[b];
            size_t decodedTargetLen = 0LU;

            // Decoding target: merge repeated characters if preprocess_collapse_repeated == True,
            // find unique elemnts if unique == True.
            // Inserts blanks before each index and a blank at the end.
            const int* target = &labels[b * maxTime];
            targetDB[b].resize(actualTargetLen * 2 + 1);
            auto& targetD = targetDB[b];
            if (unique) {
                std::unordered_set<int> uniqVals;
                for (size_t t = 0LU; t < actualTargetLen; t++) {
                    if (uniqVals.find(target[t]) != uniqVals.end()) {
                        continue;
                    }
                    uniqVals.insert(target[t]);
                    targetD[decodedTargetLen++] = blankIndex;
                    targetD[decodedTargetLen++] = target[t];
                }
                targetD[decodedTargetLen++] = blankIndex;
            } else if (preprocessCollapseRepeated) {
                auto prevValue = target[0];
                targetD[decodedTargetLen++] = blankIndex;
                targetD[decodedTargetLen++] = target[0];
                for (size_t t = 1LU; t < actualTargetLen; t++) {
                    if (target[t] == prevValue) {
                        continue;
                    }
                    targetD[decodedTargetLen++] = blankIndex;
                    targetD[decodedTargetLen++] = prevValue = target[t];
                }
                targetD[decodedTargetLen++] = blankIndex;
            } else {
                for (size_t t = 0LU; t < actualTargetLen; t++) {
                    targetD[decodedTargetLen++] = blankIndex;
                    targetD[decodedTargetLen++] = target[t];
                }
                targetD[decodedTargetLen++] = blankIndex;
            }
            decodedTargetLenB[b] = decodedTargetLen;

            auto& logProbabilities = logProbabilitiesB[b];
            logProbabilities.resize(actualLogitLen);
            for (size_t ll = 0; ll < actualLogitLen; ll++) {
                logProbabilities[ll].resize(decodedTargetLen);
            }
        }  // for batch
    };  // threadBody_1

    parallel_nt(threads_num, threadBody_1);
    if (returnCode != 0) {
        std::string resErr;
        for (auto& err : errorMsgB) {
            if (!err.empty()) {
                resErr += err + "\n";
            }
        }
        CPU_NODE_THROW(resErr);
    }

    const size_t TC = maxTime * classesNum;

    size_t workAmount2 = 0LU;
    for (size_t b = 0; b < batchNum; b++) {
        workAmount2 += logitsLength[b];
    }

    auto threadBody_2 = [&](const int ithr, const int nthr) {
        size_t start(0LU);
        size_t end(0LU);
        size_t sB(0LU);
        size_t sT(0LU);
        splitter(workAmount2, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }
        int64_t cw = 0;
        int64_t st = start;
        for (; sB < batchNum; sB++) {
            cw += logitsLength[sB];
            if (cw >= st) {
                sT = logitsLength[sB] + st - cw;
                break;
            }
        }
        size_t workCounter = start;

        for (size_t b = sB; b < batchNum; b++) {
            const size_t actualLogitLen = logitsLength[b];
            const size_t decodedTargetLen = decodedTargetLenB[b];
            auto& logProbabilities = logProbabilitiesB[b];
            auto& targetD = targetDB[b];

            double expSum = 0.0;
            size_t btcT = b * TC + sT * classesNum;
            // logProbabilities = logSoftmax = logits[b][t][c] - ln(sum_c(exp(logits[b][t])))
            for (size_t t = sT; t < actualLogitLen; t++) {
                expSum = 0.0;
                for (size_t c = 0LU; c < classesNum; c++) {
                    expSum += std::exp(logits[btcT + c]);
                }
                for (size_t s = 0LU; s < decodedTargetLen; s++) {
                    logProbabilities[t][s] = static_cast<float>(logits[btcT + targetD[s]] - std::log(expSum));
                }
                btcT += classesNum;
                if (++workCounter >= end) {
                    return;
                }
            }
            sT = 0LU;
        }  // for batch
    };  // threadBody_2

    parallel_nt(0, threadBody_2);

    const auto float_inf = std::numeric_limits<float>::infinity();

    auto sumLogs = [&float_inf](float log1, float log2) {
        if (log1 == -float_inf) {
            return log2;
        }
        if (log2 == -float_inf) {
            return log1;
        }
        if (log1 > log2) {
            return log1 + std::log1pf(std::exp(log2 - log1));
        }
        return log2 + std::log1pf(std::exp(log1 - log2));
    };

    auto threadBody_3 = [&](const int ithr, const int nthr) {
        size_t start(0LU);
        size_t end(0LU);
        splitter(batchNum, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }

        // As per Connectionist Temporal Classification - Labeling Unsegmented Sequence Data with Recurrent Neural
        // Networks: Graves et al., 2016, paragraph 4.1 (10)
        for (size_t b = start; b < end; b++) {
            auto& targetD = targetDB[b];
            auto& logProbabilities = logProbabilitiesB[b];
            const int actualLogitLen = logitsLength[b];
            const int decodedTargetLen = decodedTargetLenB[b];
            std::vector<std::vector<float>> logBwd(decodedTargetLen, std::vector<float>(actualLogitLen, -float_inf));
            for (int s = decodedTargetLen - 2; s < decodedTargetLen; s++) {
                logBwd[s][actualLogitLen - 1] = 0.F;
            }

            for (int t = actualLogitLen - 2; t >= 0; t--) {
                const int t_1 = t + 1;
                for (int s = std::max(0, decodedTargetLen - (2 * (actualLogitLen - t)));
                     s < std::min(decodedTargetLen, 2 * (t_1));
                     s++) {
                    if (ctcMergeRepeated || targetD[s] == blankIndex) {
                        logBwd[s][t] = sumLogs(logBwd[s][t], logBwd[s][t_1] + logProbabilities[t_1][s]);
                    }

                    if (s + 1 < decodedTargetLen) {
                        logBwd[s][t] = sumLogs(logBwd[s][t], logBwd[s + 1][t_1] + logProbabilities[t_1][s + 1]);
                    }

                    if (s + 2 < decodedTargetLen) {
                        if (targetD[s] != blankIndex && (!ctcMergeRepeated || (targetD[s] != targetD[s + 2]))) {
                            logBwd[s][t] = sumLogs(logBwd[s][t], logBwd[s + 2][t_1] + logProbabilities[t_1][s + 2]);
                        }
                    }
                }
            }

            logBwd[0][0] += logProbabilities[0][0];
            logBwd[1][0] += logProbabilities[0][(decodedTargetLen > 1) ? 1 : 0];

            dstData[b] = -sumLogs(logBwd[0][0], logBwd[1][0]);
        }  // for batch
    };  // threadBody_3

    parallel_nt(0, threadBody_3);
}

bool CTCLoss::created() const {
    return getType() == Type::CTCLoss;
}

}  // namespace ov::intel_cpu::node
