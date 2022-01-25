# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import numpy as np
import os
import re
import warnings
import xml.etree.ElementTree as ET
from openvino.tools.mo.utils.ir_engine.ir_engine import IREngine
from pathlib import Path

from common.constants import test_device, test_precision
from common.layer_utils import IEInfer
from common.utils.common_utils import generate_ir
from common.utils.parsers import mapping_parser


class CommonLayerTest:
    input_model_key = "input_model"

    def produce_model_path(self, framework_model, save_path):
        pass

    def get_framework_results(self, inputs_dict, model_path):
        pass

    def _test(self, framework_model, ref_net, ie_device, precision, ir_version, temp_dir, use_new_frontend=False,
              infer_timeout=60, enabled_transforms='', disabled_transforms='', **kwargs):
        """
        :param enabled_transforms/disabled_transforms: string with idxs of transforms that should be enabled/disabled.
                                                       Example: "transform_1,transform_2"
        """
        model_path = self.produce_model_path(framework_model=framework_model, save_path=temp_dir)

        self.use_new_frontend = use_new_frontend
        # TODO Pass environment variables via subprocess environment
        os.environ['MO_ENABLED_TRANSFORMS'] = enabled_transforms
        os.environ['MO_DISABLED_TRANSFORMS'] = disabled_transforms

        mo_params = {self.input_model_key: model_path,
                     "output_dir": temp_dir,
                     "data_type": precision, "model_name": 'model'
                     }

        if 'input_shapes' in kwargs and len(kwargs['input_shapes']):
            input_shapes_str = []
            for ishape in kwargs['input_shapes']:
                input_shapes_str.append('[' + ','.join([str(i) for i in ishape]) + ']')
            mo_params.update(dict(input_shape=','.join(input_shapes_str)))

        if 'input_names' in kwargs and len(kwargs['input_names']):
            mo_params.update(dict(input=','.join(kwargs['input_names'])))

        if use_new_frontend:
            mo_params["use_new_frontend"] = True

        exit_code, stderr = generate_ir(**mo_params)

        del os.environ['MO_ENABLED_TRANSFORMS']
        del os.environ['MO_DISABLED_TRANSFORMS']
        assert not exit_code, ("IR generation failed with {} exit code: {}".format(exit_code, stderr))

        path_to_xml = Path(temp_dir, 'model.xml')
        path_to_bin = Path(temp_dir, 'model.bin')

        # TODO: need to update ref graphs or get rid of this comparison
        # if ref_net is not None:
        #     ir = IREngine(path_to_xml, path_to_bin, precision=precision)
        #     (flag, resp) = ir.compare(ref_net)
        #     assert flag, '\n'.join(resp)

        from openvino.inference_engine import IECore
        core = IECore()
        net = core.read_network(path_to_xml, path_to_bin)
        inputs_info = {}
        for item in net.input_info.items():
            inputs_info[item[0]] = item[1].tensor_desc.dims

        # Prepare feed dict
        if 'kwargs_to_prepare_input' in kwargs and kwargs['kwargs_to_prepare_input']:
            inputs_dict = self._prepare_input(inputs_info, kwargs['kwargs_to_prepare_input'])
        else:
            inputs_dict = self._prepare_input(inputs_info)

        # IE infer:
        ie_engine = IEInfer(model=path_to_xml,
                            weights=path_to_bin,
                            device=ie_device)
        infer_res = ie_engine.infer(input_data=inputs_dict, infer_timeout=infer_timeout)

        if hasattr(self, 'skip_framework') and self.skip_framework:
            warnings.warn('Framework is skipped')
            return

        # Framework infer:
        fw_res = self.get_framework_results(inputs_dict=inputs_dict, model_path=model_path)

        if len(fw_res) == len(infer_res) == 1:
            # match output layers directly
            mapping_dict = {next(iter(fw_res)): next(iter(infer_res))}
        else:
            # Load mapping file
            mapping_dict = mapping_parser(path_to_xml.with_suffix('.mapping'))

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            custom_eps = kwargs['custom_eps']
        else:
            custom_eps = 1e-4

        # Compare Ie results with Framework results
        fw_eps = custom_eps if precision == 'FP32' else 5e-2
        assert self.compare_ie_results_with_framework(infer_res=infer_res, framework_res=fw_res,
                                                      mapping_dict=mapping_dict, framework_eps=fw_eps), \
            "Comparing with Framework failed: ie_res={}; framework_res={}.".format(infer_res, fw_res)

        if len(inputs_dict.keys()) > 1 or len(infer_res.keys()) > 1:
            tree = ET.parse(path_to_xml)
            # findall returns elements in document order, this order should be the same as
            # order of inputs/outputs in original model
            inputs_ie = [child for child in tree.findall('.//layer[@type="Parameter"]')]
            outputs_ie = [child for child in tree.findall('.//layer[@type="Result"]')]

            if 'input_names' in kwargs:
                input_names = kwargs['input_names']
                for i, input_name in enumerate(input_names):
                    assert inputs_ie[i].attrib['name'] == input_name, \
                        'Input order does not match framework order. Input with index {} is {}, ' \
                        'but expected {}'.format(i, inputs_ie[i].attrib['name'], input_name)

            if 'output_names' in kwargs:
                output_names = kwargs['output_names']
                for i, output_name in enumerate(output_names):
                    output_name_ie = outputs_ie[i].attrib['name']
                    output_without_sink_port = re.sub(r'\/sink_port_.', '', output_name_ie)

                    assert output_without_sink_port == output_name, \
                        'Output order does not match framework order. Output with index {} is {}, ' \
                        'but expected {}'.format(i, output_without_sink_port, output_name)


    # Feed dict for each input is filled with random number.
    # It is possible to redefine this function and generate your own input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(-255, 255, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def compare_ie_results_with_framework(self, infer_res, framework_res, mapping_dict, framework_eps):
        is_ok = True
        from common.utils.common_utils import allclose
        for framework_out_name in framework_res:

            if framework_out_name not in list(infer_res.keys()):
                if framework_out_name not in mapping_dict:
                    raise RuntimeError("Output {} not found in mapping file!".format(framework_out_name))
                ie_out_name = mapping_dict[framework_out_name]
            else:
                ie_out_name = framework_out_name

            if not allclose(infer_res[ie_out_name], framework_res[framework_out_name], atol=framework_eps,
                            rtol=framework_eps):
                is_ok = False
                print("Max diff is {}".format(
                    np.array(abs(infer_res[ie_out_name] - framework_res[framework_out_name])).max()))
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(framework_eps, framework_eps))
        return is_ok


def get_params(ie_device=None, precision=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    """

    ie_device_params = ie_device if ie_device else test_device
    precision_params = precision if precision else test_precision

    test_args = []
    for element in itertools.product(ie_device_params, precision_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def check_ir_version(left, right, ir_version):
    try:
        _ir_version = int(ir_version)
    except ValueError:
        raise RuntimeError("Wrong ir version type: {}, must be an integer".format(ir_version))
    left_bound = _ir_version - 1 if left is None else left
    right_bound = _ir_version + 1 if right is None else right
    return left_bound <= _ir_version < right_bound
