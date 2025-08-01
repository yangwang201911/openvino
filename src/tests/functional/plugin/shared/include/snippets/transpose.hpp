// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,            // Input 0 Shape
        std::vector<int>,      // Transpose order
        size_t,                // Expected num nodes
        size_t,                // Expected num subgraphs
        std::string            // Target Device
> TransposeParams;

typedef std::tuple<
        std::pair<InputShape, InputShape>, // Input Shapes
        std::vector<int>,                  // Transpose order
        size_t,                            // Expected num nodes
        size_t,                            // Expected num subgraphs
        std::string                        // Target Device
> TransposeMulParams;

class Transpose : public testing::WithParamInterface<ov::test::snippets::TransposeParams>,
                  virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeParams> obj);

protected:
    void SetUp() override;
};

class TransposeMul : public testing::WithParamInterface<ov::test::snippets::TransposeMulParams>,
                     virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMulParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov
