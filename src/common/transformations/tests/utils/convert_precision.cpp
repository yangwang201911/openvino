// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset10.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <queue>
#include <string>
#include <transformations/convert_precision.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_subgraphs.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace testing;
using namespace ngraph;

template <ngraph::element::Type_t T>
bool has_type(std::shared_ptr<ngraph::Function> f) {
    for (auto& node : f->get_ordered_ops()) {
        for (auto& input : node->inputs()) {
            if (input.get_element_type() == element::Type(T)) {
                return true;
            }
        }
        for (auto& output : node->outputs()) {
            if (output.get_element_type() == element::Type(T)) {
                return true;
            }
        }
    }
    return false;
}

TEST(TransformationTests, ConvertPrecision_NMS3) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f16, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset3::Constant::create(element::f16, Shape{}, {0.75});
        auto score_threshold = opset3::Constant::create(element::f16, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_NMS4) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f16, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f16, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_NMS5) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1000, 4});
        auto scores = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto iou_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.75});
        auto score_threshold = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::opset5::NonMaxSuppression>(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER,
            true);

        auto result1 = std::make_shared<ngraph::opset5::Result>(nms->output(0));
        auto result2 = std::make_shared<ngraph::opset5::Result>(nms->output(1));
        auto result3 = std::make_shared<ngraph::opset5::Result>(nms->output(2));
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                               ngraph::ParameterVector{boxes, scores});
    }

    pass::Manager manager;
    static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                {ngraph::element::f32, ngraph::element::f16}};
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f32>(f));
}

TEST(TransformationTests, ConvertPrecision_MatrixNms) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::Shape{1, 1000, 4});
        auto scores = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::Shape{1, 1, 1000});
        op::v8::MatrixNms::Attributes attrs;
        attrs.output_type = ngraph::element::i64;
        auto nms = std::make_shared<ngraph::opset8::MatrixNms>(boxes, scores, attrs);

        auto result1 = std::make_shared<ngraph::opset8::Result>(nms->output(0));
        auto result2 = std::make_shared<ngraph::opset8::Result>(nms->output(1));
        auto result3 = std::make_shared<ngraph::opset8::Result>(nms->output(2));
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                               ngraph::ParameterVector{boxes, scores});
    }

    pass::Manager manager;
    static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                {ngraph::element::f16, ngraph::element::f32}};
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_MulticlassNms) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto boxes = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::Shape{1, 1000, 4});
        auto scores = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::Shape{1, 1, 1000});
        op::v8::MulticlassNms::Attributes attrs;
        attrs.output_type = ngraph::element::i64;
        auto nms = std::make_shared<ngraph::opset8::MulticlassNms>(boxes, scores, attrs);

        auto result1 = std::make_shared<ngraph::opset8::Result>(nms->output(0));
        auto result2 = std::make_shared<ngraph::opset8::Result>(nms->output(1));
        auto result3 = std::make_shared<ngraph::opset8::Result>(nms->output(2));
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                               ngraph::ParameterVector{boxes, scores});
    }

    pass::Manager manager;
    static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                {ngraph::element::f16, ngraph::element::f32}};
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
    manager.run_passes(f);
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_ShapeOf) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto shape_of = std::make_shared<opset4::ShapeOf>(input);

        f = std::make_shared<Function>(NodeVector{shape_of}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_Range) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto start = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto stop = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto shift = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto range = std::make_shared<opset4::Range>(start, stop, shift, element::i64);

        f = std::make_shared<Function>(NodeVector{range}, ParameterVector{start, stop, shift});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_ConstantRelu) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = opset4::Constant::create(element::f16, Shape{1, 1000, 4}, {0});
        auto relu1 = std::make_shared<opset4::Relu>(input);
        auto relu2 = std::make_shared<opset4::Relu>(relu1);

        f = std::make_shared<Function>(NodeVector{relu2}, ParameterVector{});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_Convert) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto convert = std::make_shared<opset4::Convert>(input, element::i64);

        f = std::make_shared<Function>(NodeVector{convert}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_ConvertElimination) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);
        auto convert = std::make_shared<opset4::Convert>(relu, element::f32);

        f = std::make_shared<Function>(NodeVector{convert}, ParameterVector{input});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::f16, ngraph::element::f32}});
        manager.run_passes(f);
        ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto relu = std::make_shared<opset4::Relu>(input);

        f_ref = std::make_shared<Function>(NodeVector{relu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertPrecision_TopK) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i64);

        f = std::make_shared<Function>(OutputVector{topk->output(0), topk->output(1)}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Unique10) {
    std::shared_ptr<ov::Model> model(nullptr);
    {
        auto input = std::make_shared<op::Parameter>(element::f16, Shape{15, 20, 3});
        auto unique = std::make_shared<opset10::Unique>(input);

        model = std::make_shared<ov::Model>(unique->outputs(), ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{element::i64, element::i32}, {element::f16, element::f32}};

        manager.register_pass<pass::ConvertPrecision>(precisions);
        manager.run_passes(model);
    }

    ASSERT_EQ(model->outputs().size(), 4);
    EXPECT_EQ(model->outputs()[0].get_element_type(), element::f32);
    EXPECT_EQ(model->outputs()[1].get_element_type(), element::i32);
    EXPECT_EQ(model->outputs()[2].get_element_type(), element::i32);
    EXPECT_EQ(model->outputs()[3].get_element_type(), element::i32);

    EXPECT_EQ(model->get_results().size(), 4);

    EXPECT_FALSE(has_type<element::Type_t::f16>(model));
    EXPECT_FALSE(has_type<element::Type_t::i64>(model));
}

TEST(TransformationTests, ConvertPrecision_NonZero) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto non_zero = std::make_shared<ngraph::opset4::NonZero>(input, ngraph::element::i64);

        f = std::make_shared<Function>(OutputVector{non_zero}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Bucketize) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{20});
        auto k = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        auto b = std::make_shared<ngraph::opset4::Bucketize>(input, k);

        f = std::make_shared<Function>(OutputVector{b}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_Roundings) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto max_int64 = std::numeric_limits<int64_t>::max();
        auto max_int32 = std::numeric_limits<int32_t>::max();

        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{5, 5, 5, 5});
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64,
                                                    ngraph::Shape{4},
                                                    {max_int64, max_int64, max_int64, max_int64});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(input, begin, end, stride, begin_mask, end_mask);

        f = std::make_shared<Function>(OutputVector{ss}, ParameterVector{input});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);

        auto casted_end = std::dynamic_pointer_cast<opset1::Constant>(ss->input_value(2).get_node_shared_ptr());
        ASSERT_TRUE(casted_end != nullptr);
        ASSERT_EQ(casted_end->get_element_type(), element::i32);
        ASSERT_EQ(casted_end->cast_vector<int32_t>(),
                  std::vector<int32_t>({max_int32, max_int32, max_int32, max_int32}));
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_TIBody) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f16, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f16, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset4::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(gru_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(gru_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::i64, ngraph::element::i32},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);

        ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
        ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(f));
        ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(tensor_iterator->get_body()));
        ASSERT_FALSE(has_type<ngraph::element::Type_t::i64>(tensor_iterator->get_body()));
    }
}

TEST(TransformationTests, ConvertPrecision_Equal) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::Equal>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::boolean, ngraph::element::u8},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_NotEqual) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::NotEqual>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::boolean, ngraph::element::u8},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_Greater) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::Greater>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::boolean, ngraph::element::u8},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_GreaterEqual) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::GreaterEqual>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::boolean, ngraph::element::u8},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_Less) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::Less>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::boolean, ngraph::element::u8},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LessEqual) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::LessEqual>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;

        static const precisions_array precisions = {{ngraph::element::boolean, ngraph::element::u8},
                                                    {ngraph::element::f16, ngraph::element::f32}};

        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalAnd) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::LogicalAnd>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::boolean, ngraph::element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalOr) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::LogicalOr>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::boolean, ngraph::element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalXor) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::LogicalXor>(input1, input2);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::boolean, ngraph::element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_LogicalNot) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::LogicalNot>(input1);

        f = std::make_shared<Function>(OutputVector{node}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::boolean, ngraph::element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_Select) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::LogicalNot>(input1);
        auto select = std::make_shared<ngraph::opset4::Select>(node, input1, input1);

        f = std::make_shared<Function>(OutputVector{select}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::boolean, ngraph::element::u8}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::u8>(f));
}

TEST(TransformationTests, ConvertPrecision_TypeRelaxedWithSelect) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ngraph::opset4::LogicalNot>(input1);
        auto select = std::make_shared<ngraph::opset4::Select>(node, input1, input1);

        f = std::make_shared<Function>(OutputVector{select}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::boolean, ngraph::element::i32}});
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::i32, ngraph::element::i64}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
    ASSERT_FALSE(has_type<ngraph::element::Type_t::i32>(f));
    ASSERT_TRUE(has_type<ngraph::element::Type_t::i64>(f));
}

TEST(TransformationTests, ConvertPrecision_TypeRelaxed) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::boolean, ngraph::Shape{15, 20, 3});
        auto select = std::make_shared<ngraph::opset4::Select>(input1, input1, input1);
        auto type_relaxed = std::make_shared<op::TypeRelaxed<opset4::Select>>(*select,
                                                                              element::TypeVector{},
                                                                              element::TypeVector{element::i64});

        f = std::make_shared<Function>(OutputVector{type_relaxed}, ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::boolean, ngraph::element::i32}});
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::i32, ngraph::element::i64}});
        manager.run_passes(f);

        ASSERT_FALSE(has_type<ngraph::element::Type_t::boolean>(f));
        ASSERT_FALSE(has_type<ngraph::element::Type_t::i32>(f));
        ASSERT_TRUE(has_type<ngraph::element::Type_t::i64>(f));
    }
}

TEST(TransformationTests, ConvertPrecision_Variables) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        Shape shape{1, 10, 2};
        auto inp = std::make_shared<opset4::Parameter>(element::f16, shape);
        auto m_i = std::make_shared<opset4::Constant>(element::f16, shape, 1);
        auto m_r = std::make_shared<opset4::ReadValue>(m_i, "ID");
        auto sum = std::make_shared<opset4::Add>(inp, m_r);
        auto m_w = std::make_shared<opset4::Assign>(sum, "ID");
        auto mul = std::make_shared<opset4::Multiply>(inp, sum);

        mul->add_control_dependency(m_w);

        f = std::make_shared<Function>(NodeVector{mul}, ParameterVector{inp});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::f16, ngraph::element::f32}});
        manager.run_passes(f);
    }

    ASSERT_FALSE(has_type<ngraph::element::Type_t::f16>(f));
}

TEST(TransformationTests, ConvertPrecision_skip_precision_sensitive) {
    std::shared_ptr<ov::Model> model(nullptr);
    std::shared_ptr<opset8::Interpolate> interpolate(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto sizes = opset8::Constant::create(element::i64, Shape{4}, {1, 3, 288, 512});
        auto scales = opset8::Constant::create(element::f32, Shape{4}, {1.0f, 1.0f, 0.4f, 0.4f});
        opset8::Interpolate::InterpolateAttrs attrs;

        attrs.mode = opset8::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = opset8::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        interpolate = std::make_shared<opset8::Interpolate>(input, sizes, scales, attrs);
        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});

        pass::Manager manager;

        manager.register_pass<ov::pass::MarkPrecisionSensitiveSubgraphs>();
        manager.get_pass_config()->set_callback<ngraph::pass::ConvertPrecision>(
            [](const std::shared_ptr<const Node>& node) -> bool {
                return ov::fp16_compression_is_disabled(node) && node->get_element_type() == element::f32;
            });

        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}});
        manager.run_passes(model);
    }

    ASSERT_TRUE(has_type<element::Type_t::f32>(model));
    ASSERT_TRUE(interpolate->input_value(2).get_element_type() == element::Type_t::f32);
}

TEST(TransformationTests, ConvertPrecision_without_callback) {
    // without callback all nodes should be converted to f16 regardless even they are marked
    std::shared_ptr<ov::Model> model(nullptr);
    std::shared_ptr<opset8::Interpolate> interpolate(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto sizes = opset8::Constant::create(element::i64, Shape{4}, {1, 3, 288, 512});
        auto scales = opset8::Constant::create(element::f32, Shape{4}, {1.0f, 1.0f, 0.4f, 0.4f});
        opset8::Interpolate::InterpolateAttrs attrs;

        attrs.mode = opset8::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = opset8::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        interpolate = std::make_shared<opset8::Interpolate>(input, sizes, scales, attrs);
        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        pass::Manager manager;

        manager.register_pass<ov::pass::MarkPrecisionSensitiveSubgraphs>();
        manager.register_pass<pass::ConvertPrecision>(precisions_array{{element::f32, element::f16}});
        manager.run_passes(model);
    }

    ASSERT_FALSE(has_type<element::Type_t::f32>(model));
    ASSERT_TRUE(interpolate->input_value(2).get_element_type() == element::Type_t::f16);
}

template <typename From, typename To>
void constant_convert_test(element::Type type_from,
                           element::Type type_to,
                           const std::vector<From>& value,
                           const std::vector<To>& expected) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    std::string expected_friendly_name;
    size_t size = value.size() * sizeof(From) * 8 / type_from.bitwidth();
    {
        auto c = std::make_shared<opset4::Constant>(type_from, Shape{size}, value.data());
        expected_friendly_name = c->get_friendly_name();
        f = std::make_shared<Function>(NodeVector{c}, ParameterVector{});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array{{type_from, type_to}});
        manager.run_passes(f);
    }
    auto ops = f->get_ordered_ops();
    auto c = std::dynamic_pointer_cast<opset4::Constant>(ops[0]);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(c->get_friendly_name(), expected_friendly_name);
    std::vector<To> actual;
    try {
        actual = c->cast_vector<To>();
    } catch (...) {
        size_t dst_size = (type_to.bitwidth() * size + 7) / 8;
        actual.assign(c->get_data_ptr<uint8_t>(), c->get_data_ptr<uint8_t>() + dst_size);
    }
    ASSERT_TRUE(actual.size() >= expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(expected[i], actual[i]);
    }
}

template <typename From, typename To>
void constant_convert_test(element::Type_t type_from, element::Type_t type_to, From value, To expected) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    std::string expected_friendly_name;
    {
        auto c = std::make_shared<opset4::Constant>(type_from, Shape{}, &value);
        expected_friendly_name = c->get_friendly_name();
        f = std::make_shared<Function>(NodeVector{c}, ParameterVector{});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array{{type_from, type_to}});
        manager.run_passes(f);
    }
    auto ops = f->get_ordered_ops();
    auto c = std::dynamic_pointer_cast<opset4::Constant>(ops[0]);
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(c->get_friendly_name(), expected_friendly_name);

    auto actual = c->cast_vector<To>()[0];
    ASSERT_EQ(expected, actual);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I64MinToI32) {
    constant_convert_test(element::Type_t::i64,
                          element::Type_t::i32,
                          std::numeric_limits<int64_t>::min(),
                          std::numeric_limits<int32_t>::min());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I64MaxToI32) {
    constant_convert_test(element::Type_t::i64,
                          element::Type_t::i32,
                          std::numeric_limits<int64_t>::max(),
                          std::numeric_limits<int32_t>::max());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U64MinToI32) {
    constant_convert_test(element::Type_t::u64, element::Type_t::i32, std::numeric_limits<uint64_t>::min(), 0);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U64MaxToI32) {
    constant_convert_test(element::Type_t::u64,
                          element::Type_t::i32,
                          std::numeric_limits<uint64_t>::max(),
                          std::numeric_limits<int32_t>::max());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U64ToI32) {
    constant_convert_test<uint64_t, int32_t>(element::Type_t::u64, element::Type_t::i32, 42, 42);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U32MinToI32) {
    constant_convert_test(element::Type_t::u32, element::Type_t::i32, std::numeric_limits<uint32_t>::min(), 0);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U32MaxToI32) {
    constant_convert_test(element::Type_t::u32,
                          element::Type_t::i32,
                          std::numeric_limits<uint32_t>::max(),
                          std::numeric_limits<int32_t>::max());
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U32ToI32) {
    constant_convert_test(element::Type_t::u32, element::Type_t::i32, 42, 42);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_BoolToU8) {
    constant_convert_test(element::Type_t::boolean, element::Type_t::u8, true, 1);
    constant_convert_test(element::Type_t::boolean, element::Type_t::u8, false, 0);
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI8) {
    constant_convert_test<uint8_t, int8_t>(element::u4, element::i8, std::vector<uint8_t>{171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU8) {
    constant_convert_test<uint8_t, uint8_t>(element::u4, element::u8, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI8_2) {
    constant_convert_test<uint8_t, int8_t>(element::u4, element::i8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU8_96) {
    constant_convert_test<uint8_t, uint8_t>(element::u4, element::u8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU8) {
    constant_convert_test<uint8_t, uint8_t>(element::i4, element::u8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI8) {
    constant_convert_test<uint8_t, int8_t>(element::i4, element::i8, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU8_neg) {
    constant_convert_test<uint8_t, uint8_t>(element::i4, element::u8, {171}, {250, 251});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI8_neg) {
    constant_convert_test<uint8_t, int8_t>(element::i4, element::i8, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI32) {
    constant_convert_test<uint8_t, int32_t>(element::u4, element::i32, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU32) {
    constant_convert_test<uint8_t, uint32_t>(element::u4, element::u32, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU32) {
    constant_convert_test<uint8_t, uint32_t>(element::i4, element::u32, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI32) {
    constant_convert_test<uint8_t, int32_t>(element::i4, element::i32, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU32_neg) {
    constant_convert_test<uint8_t, uint32_t>(element::i4, element::u32, {171}, {4294967290, 4294967291});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI32_neg) {
    constant_convert_test<uint8_t, int32_t>(element::i4, element::i32, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI16) {
    constant_convert_test<uint8_t, int16_t>(element::u4, element::i16, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU16) {
    constant_convert_test<uint8_t, uint16_t>(element::u4, element::u16, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU16) {
    constant_convert_test<uint8_t, uint16_t>(element::i4, element::u16, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI16) {
    constant_convert_test<uint8_t, int16_t>(element::i4, element::i16, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU16_neg) {
    constant_convert_test<uint8_t, uint16_t>(element::i4, element::u16, {171}, {65530, 65531});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI16_neg) {
    constant_convert_test<uint8_t, int16_t>(element::i4, element::i16, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToI64) {
    constant_convert_test<uint8_t, int64_t>(element::u4, element::i64, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U4ToU64) {
    constant_convert_test<uint8_t, int64_t>(element::u4, element::u64, {171}, {10, 11});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU64) {
    constant_convert_test<uint8_t, uint64_t>(element::i4, element::u64, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI64) {
    constant_convert_test<uint8_t, int64_t>(element::i4, element::i64, {96}, {6, 0});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToU64_neg) {
    constant_convert_test<uint8_t, uint64_t>(element::i4,
                                             element::u64,
                                             {171},
                                             {18446744073709551610u, 18446744073709551611u});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_I4ToI64_neg) {
    constant_convert_test<uint8_t, int64_t>(element::i4, element::i64, {171}, {-6, -5});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U1ToU8) {
    constant_convert_test<uint8_t, uint8_t>(element::u1, element::u8, {171}, {1, 0, 1, 0, 1, 0, 1, 1});
}

TEST(TransformationTests, ConvertPrecision_ConstantConversion_U1ToU4) {
    constant_convert_test<uint8_t, uint8_t>(element::u1,
                                            element::u4,
                                            std::vector<uint8_t>{171},
                                            {1, 0, 1, 0, 1, 0, 1, 1});
}
