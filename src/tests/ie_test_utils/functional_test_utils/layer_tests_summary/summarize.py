# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import csv
import defusedxml.ElementTree as ET
from defusedxml import defuse_stdlib

from jinja2 import Environment, FileSystemLoader

from utils.conformance_utils import get_logger
from utils import stat_update_utils

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ET_defused = defuse_stdlib()[ET]
Element = ET_defused.Element
SubElement = ET_defused.SubElement

NOT_RUN = "NOT RUN"
NA = "N/A"

STATUS_CSV_ORDER = ["implemented", "passed", "failed", "skipped", "crashed", "hanged", "passrate", "relative_passrate"]

logger = get_logger('conformance_summary')


def parse_arguments():
    parser = argparse.ArgumentParser()

    xml_help = """
        Paths to xml summary files from layer tests.
        In case of entries intersection, results will
        be merged basing on timestamp - entry from latest
        report is be kept.
    """
    out_help = "Path where to save html report"
    report_tag = "Report tag"
    report_version = "Report version"
    output_filename_help = "Output report filename"
    conformance_mode_help = "Allow to align test number"
    csv_help = "Allow to serialize report as csv file"
    expected_devices_help = "List of expected devices"

    parser.add_argument("--xml", help=xml_help, nargs="*", required=True)
    parser.add_argument("--out", help=out_help, default="")
    parser.add_argument("--output_filename", help=output_filename_help, default="report")
    parser.add_argument("--report_tag", help=report_tag, default="")
    parser.add_argument("--report_version", help=report_version, default="")
    parser.add_argument("--conformance_mode", help=conformance_mode_help, default=False)
    parser.add_argument("--csv", help=csv_help, default=False)
    parser.add_argument("--expected_devices", help=expected_devices_help, nargs="*", required=False)

    return parser.parse_args()


def merge_xmls(xml_paths: list):
    logger.info("Merging XML files is started")

    summary = Element("report")
    timestamp = None
    summary_results = SubElement(summary, "results")
    ops_list = SubElement(summary, "ops_list")
    for xml_path in xml_paths:
        try:
            xml_root = ET.parse(xml_path).getroot()
            logger.info(f'Info from {xml_path} is adding to the final summary')
        except ET.ParseError:
            logger.error(f'Error parsing {xml_path}')

        if timestamp is None or timestamp < xml_root.attrib["timestamp"]:
            logger.info(f'Timestamp is updated from {timestamp} to {xml_root.attrib["timestamp"]}')
            timestamp = xml_root.attrib["timestamp"]

        for op in xml_root.find("ops_list"):
            if ops_list.find(op.tag) is None:
                SubElement(ops_list, op.tag)

        for device in xml_root.find("results"):
            device_results = summary_results.find(device.tag)
            if device_results is None:
                summary_results.append(device)
            else:
                for op_result in device:
                    current_op_res = device_results.find(op_result.tag)
                    if current_op_res is not None:
                        # workaround for unsaved reports
                        total_tests_count_xml, total_tests_count_summary = (0, 0)
                        for attr_name in device_results.find(op_result.tag).attrib:
                            if "relative_" in attr_name or attr_name == "passrate" or attr_name == "implemented":
                                continue
                            total_tests_count_xml += int(op_result.attrib.get(attr_name))
                            total_tests_count_summary += int(current_op_res.attrib.get(attr_name))
                        if total_tests_count_xml > total_tests_count_summary:
                            logger.warning(f'Test counter is different in {op_result.tag} for {device.tag}'\
                                           f'({total_tests_count_xml} vs {total_tests_count_xml})')
                            for attr_name in device_results.find(op_result.tag).attrib:
                                if attr_name == "passrate" or attr_name == "implemented":
                                    continue
                                xml_value = int(op_result.attrib.get(attr_name))
                                device_results.find(current_op_res.tag).set(attr_name, str(xml_value))
                    else:
                        device_results.append(op_result)
    stat_update_utils.update_passrates(summary_results)
    summary.set("timestamp", timestamp)
    logger.info("Merging XML files is competed")
    return summary


def collect_statistic(root: Element, is_conformance_mode: bool):
    logger.info("Statistic collecting is started")
    trusted_ops = dict()
    pass_rate_avg = dict()
    pass_rate_avg_rel = dict()
    general_pass_rate = dict()
    general_pass_rate_rel = dict()
    general_test_count = dict()
    general_test_count_rel = dict()
    general_passed_tests = dict()
    general_passed_tests_rel = dict()
    op_res = dict()

    results = dict()
    covered_ops = dict()
    for device in root.find("results"):
        results[device.tag] = {op.tag: op.attrib for op in device}

        pass_rate_avg[device.tag] = 0
        pass_rate_avg_rel[device.tag] = 0
        general_test_count[device.tag] = 0
        general_test_count_rel[device.tag] = 0
        general_passed_tests[device.tag] = 0
        general_passed_tests_rel[device.tag] = 0
        trusted_ops[device.tag] = 0
        covered_ops[device.tag] = 0
        for op in results[device.tag]:
            # for correct display of reports without hanged item in report.xml
            results[device.tag][op]["hanged"] = results[device.tag][op].get("hanged", 0)
            op_test_cnt = int(results[device.tag][op]["passed"]) + int(results[device.tag][op]["failed"]) + \
                          int(results[device.tag][op]["crashed"]) + int(results[device.tag][op]["skipped"]) + \
                          int(results[device.tag][op]["hanged"])
            if op_test_cnt == 0:
                continue
            covered_ops[device.tag] += 1
            pass_rate = float("%.2f"%float(results[device.tag][op]["passrate"]))
            relative_pass_rate = float("%.2f"%float(results[device.tag][op]["relative_passrate"]))
            results[device.tag][op]["passrate"] = pass_rate
            results[device.tag][op]["relative_passrate"] = relative_pass_rate

            if pass_rate == 100.:
                trusted_ops[device.tag] += 1
            device_general_test_count = op_test_cnt
            general_test_count[device.tag] += device_general_test_count
            general_test_count_rel[device.tag] += float(results[device.tag][op]["relative_all"])
            general_passed_tests[device.tag] += int(results[device.tag][op]["passed"])
            general_passed_tests_rel[device.tag] += float(results[device.tag][op]["relative_passed"])
            pass_rate_avg[device.tag] += float(results[device.tag][op]["passrate"])
            pass_rate_avg_rel[device.tag] += float(results[device.tag][op]["relative_passrate"])

            if op in op_res.keys():
                op_res[op].update({device.tag: device_general_test_count})
            else:
                op_res.update({op: {device.tag: device_general_test_count}})
        pass_rate_avg[device.tag] = 0 if covered_ops[device.tag] == 0 else pass_rate_avg[device.tag] / covered_ops[device.tag]
        pass_rate_avg[device.tag] = float("%.2f"%float(pass_rate_avg[device.tag]))
        pass_rate_avg_rel[device.tag] = 0 if covered_ops[device.tag] == 0 else pass_rate_avg_rel[device.tag] / covered_ops[device.tag]
        pass_rate_avg_rel[device.tag] = float("%.2f"%float(pass_rate_avg_rel[device.tag]))
        general_pass_rate[device.tag] = 0 if general_test_count[device.tag] == 0 else (general_passed_tests[device.tag] * 100 / general_test_count[device.tag])
        general_pass_rate[device.tag] = float("%.2f"%float(general_pass_rate[device.tag]))
        general_pass_rate_rel[device.tag] = 0 if general_test_count_rel[device.tag] == 0 else (general_passed_tests_rel[device.tag] * 100 / general_test_count_rel[device.tag])
        general_pass_rate_rel[device.tag] = float("%.2f"%float(general_pass_rate_rel[device.tag]))
        trusted_ops[device.tag] = float("%.2f"%(float("%.2f"%(float(trusted_ops[device.tag]) * 100)) / covered_ops[device.tag])) if device.tag in covered_ops and covered_ops[device.tag] != 0 else 0

    logger.info("Test number comparison between devices is started")
    for op in op_res:
        op_counter = None
        is_not_printed = True
        max_test_cnt = 0
        for dev in op_res[op]:
            if op_counter is None:
                op_counter = op_res[op][dev]
            elif op_counter != op_res[op][dev]:
                max_test_cnt = max(max_test_cnt, op_res[op][dev])
                if is_not_printed:
                    is_not_printed = False
                    logger.warning(f'{op} : {op_res[op]}')

    logger.info("Test number comparison between devices is completed")

    devices = results.keys()
    logger.info("Statistic collecting is completed")
    return devices, results, general_pass_rate, general_pass_rate_rel, pass_rate_avg, pass_rate_avg_rel, general_test_count, trusted_ops, covered_ops


def format_string(input_str: str):
    res = input_str
    res = res.replace('{', '')
    res = res.replace('}', '')
    res = res.replace("'", '')
    res = res.replace('"', '')
    res = res.replace(': ', '=')
    res = res.replace(' ', '')
    res = res.replace(',', ' ')
    return res


def serialize_to_csv(report_filename: str, output_dir: os.path, op_list: list, device_list: list, results: dict):
    csv_filename = os.path.join(output_dir, report_filename + '.csv')
    with open(csv_filename, "w", newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, dialect='excel')
        # csv_writer.writerow(['Operation'] + device_list)
        devices_csv = ['Operation']
        device_res_csv = ['Operation']
        
        for device in device_list:          
            for status in STATUS_CSV_ORDER:
                devices_csv.append(device)
                device_res_csv.append(status)
            
        csv_writer.writerow(devices_csv)
        csv_writer.writerow(device_res_csv)

        for op in op_list:
            list_to_csv = list()
            for device in device_list:
                if op in results[device]:
                    if results[device][op] == NA or results[device][op] == NOT_RUN:
                        for status in STATUS_CSV_ORDER:
                            list_to_csv.append(results[device][op])
                        continue
                    for status in STATUS_CSV_ORDER:
                        list_to_csv.append(str(results[device][op][status]))
                else:
                    for status in STATUS_CSV_ORDER:
                        list_to_csv.append(NA)
            csv_writer.writerow([op] + list_to_csv)

    logger.info(f'Final CSV report is saved to {csv_filename}')


def create_summary(summary_root: Element, output_folder: os.path, expected_devices:list, report_tag: str, report_version: str,
                   is_conformance_mode: bool,  is_serialize_to_csv: bool, output_filename='report'):
    if is_conformance_mode:
        stat_update_utils.update_conformance_test_counters(summary_root)
        stat_update_utils.update_passrates(summary_root.find("results"))
    device_list, results, general_pass_rate, general_pass_rate_rel, pass_rate_avg, pass_rate_avg_rel, general_test_count, trusted_ops, covered_ops = \
        collect_statistic(summary_root, is_conformance_mode)

    op_list = list()
    for op in summary_root.find("ops_list"):
        op_list.append(op.tag)
    op_list = sorted(op_list)
    
    if len(expected_devices) > 0 and sorted(expected_devices) != device_list:
        for expected_device in expected_devices:
            if expected_device in device_list:
                continue
            tmp_res = dict()
            no_run_val = "NOT RUN"
            tmp_res = {op: no_run_val for op in op_list}
            results[expected_device] = tmp_res
            general_pass_rate[expected_device] = no_run_val
            pass_rate_avg[expected_device] = no_run_val
            general_test_count[expected_device] = no_run_val
            trusted_ops[expected_device] = no_run_val
            covered_ops[expected_device] = no_run_val
        device_list = results.keys()

    timestamp = summary_root.attrib["timestamp"]

    device_list = sorted(device_list)

    script_dir, script_name = os.path.split(os.path.abspath(__file__))
    file_loader = FileSystemLoader(os.path.join(script_dir, 'template'))
    env = Environment(loader=file_loader)
    template = env.get_template('report_template.html')

    res_summary = template.render(ordered_ops=op_list, devices=device_list, results=results, timestamp=timestamp,
                                  general_pass_rate=general_pass_rate, general_pass_rate_rel=general_pass_rate_rel,
                                  pass_rate_avg=pass_rate_avg, pass_rate_avg_rel=pass_rate_avg_rel,
                                  trusted_ops=trusted_ops, covered_ops=covered_ops,
                                  general_test_count=general_test_count, report_tag=report_tag, report_version=report_version)

    report_path = os.path.join(output_folder, f'{output_filename}.html')
    with open(report_path, "w") as f:
        logger.info(f'Final report is saved to {report_path}')
        f.write(res_summary)
    if is_serialize_to_csv:
        serialize_to_csv(output_filename, output_folder, op_list, device_list, results)


if __name__ == "__main__":
    args = parse_arguments()
    summary_root = merge_xmls(args.xml)
    create_summary(summary_root, args.out,
                   [] if args.expected_devices is None else args.expected_devices,
                   args.report_tag,
                   args.report_version,
                   args.conformance_mode,
                   args.csv,
                   args.output_filename)
    
