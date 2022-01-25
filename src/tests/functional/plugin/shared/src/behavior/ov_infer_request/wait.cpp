// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/wait.hpp"
#include "openvino/runtime/exception.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestWaitTests::getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj) {
    return OVInferRequestTests::getTestCaseName(obj);
}

void OVInferRequestWaitTests::SetUp() {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OVInferRequestTests::SetUp();
    req = execNet.create_infer_request();
    input = execNet.input();
    output = execNet.output();
}

void OVInferRequestWaitTests::TearDown() {
    req = {};
    input = {};
    output = {};
    OVInferRequestTests::TearDown();
}

TEST_P(OVInferRequestWaitTests, CorrectOneAsyncInferWithGetInOutWithInfWait) {
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(output));
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(OVInferRequestWaitTests, canstart_asyncInferWithGetInOutWithStatusOnlyWait) {
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait_for({}));
}

TEST_P(OVInferRequestWaitTests, canWaitWithotStartSsync) {
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(req.wait_for({}));
    OV_ASSERT_NO_THROW(req.wait_for(std::chrono::milliseconds{1}));
}

TEST_P(OVInferRequestWaitTests, throwExceptionOnSetTensorAfterAsyncInfer) {
    auto&& config = configuration;
    auto itConfig = config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
    if (itConfig != config.end()) {
        if (itConfig->second != "CPU_THROUGHPUT_AUTO") {
            if (std::stoi(itConfig->second) == 0) {
                GTEST_SKIP() << "Not applicable with disabled streams";
            }
        }
    }
    auto output_tensor = req.get_tensor(input);
    OV_ASSERT_NO_THROW(req.wait_for({}));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(try {
        req.set_tensor(input, output_tensor);
    } catch (const ov::Busy&) {});
    OV_ASSERT_NO_THROW(req.wait_for({}));
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestWaitTests, throwExceptionOnGetTensorAfterAsyncInfer) {
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(try {
        req.get_tensor(input);
    } catch (const ov::Busy&) {});
    OV_ASSERT_NO_THROW(req.wait());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
