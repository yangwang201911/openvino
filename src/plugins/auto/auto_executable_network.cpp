// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "ie_performance_hints.hpp"
#include "auto_executable_network.hpp"

// ------------------------------AutoExecutableNetwork----------------------------
//
namespace MultiDevicePlugin {
AutoExecutableNetwork::AutoExecutableNetwork(AutoScheduleContext::Ptr& context, const AutoSchedule::Ptr& schedule)
    :ExecutableNetwork(schedule, context),
     _autoSContext(context),
     _autoSchedule(schedule) {
}

std::shared_ptr<IE::RemoteContext> AutoExecutableNetwork::GetContext() const {
    _autoSchedule->WaitActualNetworkReady();
    return _autoSchedule->_loadContext[ACTUALDEVICE].executableNetwork->GetContext();
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, IE::Parameter>
    & config) {
    IE_THROW(NotImplemented);
}

IE::Parameter AutoExecutableNetwork::GetConfig(const std::string& name) const {
    IE_THROW(NotImplemented);
}

IE::Parameter AutoExecutableNetwork::GetMetric(const std::string& name, const std::string& target_device) const {
    if (target_device == _autoSchedule->_loadContext[ACTUALDEVICE].deviceInfo.deviceName) {
        if (_autoSchedule->_loadContext[ACTUALDEVICE].future.valid()) {
            _autoSchedule->_loadContext[ACTUALDEVICE].future.wait();
        }
        try {
            return _autoSchedule->_loadContext[ACTUALDEVICE].executableNetwork->GetMetric(name);
        } catch (...) {
            return _autoSchedule->_loadContext[ACTUALDEVICE].executableNetwork->GetConfig(name);
        }
    }
    if (target_device == "CPU" && _autoSchedule->_loadContext[CPU].isAlready) {
        try {
            return _autoSchedule->_loadContext[CPU].executableNetwork->GetMetric(name);
        } catch (...) {
            return _autoSchedule->_loadContext[CPU].executableNetwork->GetConfig(name);
        }
    }
    auto actualDevice = _autoSchedule->_loadContext[ACTUALDEVICE].deviceInfo.deviceName;
    IE_THROW(NotFound) << target_device << "is not the device selected by " << GetLogTag()
                       << ". Actual selected device is " << actualDevice;
}

IE::Parameter AutoExecutableNetwork::GetMetric(const std::string& name) const {
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
            // Metrics
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::hint::performance_mode.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::model_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::hint::model_priority.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::priorities.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::execution_devices.name(), ov::PropertyMutability::RO}};
    } else if (name == ov::hint::performance_mode) {
        auto value = _autoSContext->_performanceHint;
        if (!_autoSContext->_core->isNewAPI())
            return value;
        if (value == InferenceEngine::PluginConfigParams::THROUGHPUT)
            return ov::hint::PerformanceMode::THROUGHPUT;
        else if (value == InferenceEngine::PluginConfigParams::LATENCY)
            return ov::hint::PerformanceMode::LATENCY;
        else if (value == InferenceEngine::PluginConfigParams::CUMULATIVE_THROUGHPUT)
            return ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
        else
            return ov::hint::PerformanceMode::UNDEFINED;
    } else if (name == ov::model_name) {
        if (_autoSchedule->_loadContext[ACTUALDEVICE].isAlready) {
            return _autoSchedule->_loadContext[ACTUALDEVICE].executableNetwork->GetMetric(
                    name);
        }
        return _autoSchedule->_loadContext[CPU].executableNetwork->GetMetric(name);
    } else if (name == ov::device::priorities) {
        auto value = _autoSContext->_config.find(ov::device::priorities.name());
        return decltype(ov::device::priorities)::value_type {value->second.as<std::string>()};
    } else if (name == ov::hint::model_priority) {
        auto value = _autoSContext->_modelPriority;
        if (_autoSContext->_core->isNewAPI()) {
            return value ? ((value > 1) ? ov::hint::Priority::LOW :
                    ov::hint::Priority::MEDIUM) : ov::hint::Priority::HIGH;
        } else {
            return value ? ((value > 1) ? CONFIG_VALUE(MODEL_PRIORITY_LOW) : CONFIG_VALUE(
                        MODEL_PRIORITY_MED)) : CONFIG_VALUE(MODEL_PRIORITY_HIGH);
        }
    } else if (name == ov::optimal_number_of_infer_requests) {
        const unsigned int defaultNumForTPUT = 4u;
        const unsigned int defaultNumForLatency = 1u;
        unsigned int real = 0;
        if (_autoSchedule->_loadContext[ACTUALDEVICE].isAlready) {
            real = _autoSchedule->_loadContext[ACTUALDEVICE].
                executableNetwork->GetMetric(name).as<unsigned int>();
        } else {
            IE_ASSERT(_autoSchedule->_loadContext[CPU].isAlready == true);
            std::unique_lock<std::mutex> lock(_autoSContext->_confMutex);
            auto deviceInfo = _autoSchedule->_loadContext[ACTUALDEVICE].deviceInfo;
            lock.unlock();
            unsigned int optimalBatchSize = 0;
            unsigned int requests = 0;
            bool bThroughputEnabledInPlugin = false;
            try {
                // for benchmark through AUTO:CPU,GPU
                // SetConfig directly set to CPU/GPU in this case
                bThroughputEnabledInPlugin =
                    _autoSContext->_core->GetConfig(deviceInfo.deviceName,
                        CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>() == CONFIG_VALUE(THROUGHPUT);
            } catch (...) {
                LOG_DEBUG_TAG("GetMetric:%s for %s", "PERF_HINT config not supported",
                    deviceInfo.deviceName.c_str());
            }
            const auto& mode = deviceInfo.config.find(CONFIG_KEY(PERFORMANCE_HINT));
            if (bThroughputEnabledInPlugin ||
                (mode != deviceInfo.config.end() && mode->second == CONFIG_VALUE(THROUGHPUT))) {
                unsigned int upperBoundStreamsNum = 0;
                std::map<std::string, IE::Parameter> options;
                options["MODEL_PTR"] = std::const_pointer_cast<ngraph::Function>
                    (_autoSContext->_network.getFunction());
                try {
                    auto rangeOfStreams = _autoSContext->_core->GetMetric(deviceInfo.deviceName,
                            METRIC_KEY(RANGE_FOR_STREAMS),
                            options).as<std::tuple<unsigned int, unsigned int>>();
                    upperBoundStreamsNum = std::get<1>(rangeOfStreams);
                } catch (const IE::Exception&) {
                    LOG_DEBUG_TAG("GetMetric RANGE_FOR_STREAMS failed");
                }
                if (!_autoSContext->_batchingDisabled) {
                    try {
                        optimalBatchSize = _autoSContext->_core->GetMetric(deviceInfo.deviceName,
                                METRIC_KEY(OPTIMAL_BATCH_SIZE), options).as<unsigned int>();
                        LOG_DEBUG_TAG("BATCHING:%s:%ld", "optimal batch size",
                            optimalBatchSize);
                    } catch (...) {
                        LOG_DEBUG_TAG("BATCHING:%s", "metric OPTIMAL_BATCH_SIZE not supported");
                    }
                }
                if (optimalBatchSize > 1) {
                    // batching is supported with the device
                    // go with auto-batching
                    try {
                        // check if app have set preferred value
                        auto res =
                            _autoSContext->_core->GetConfig(deviceInfo.deviceName,
                                CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
                        requests = IE::PerfHintsConfig::CheckPerformanceHintRequestValue(res);
                        const auto& reqs = deviceInfo.config.find(CONFIG_KEY(
                                    PERFORMANCE_HINT_NUM_REQUESTS));
                        if (reqs != deviceInfo.config.end()) {
                            requests = static_cast<unsigned int>
                                (IE::PerfHintsConfig::CheckPerformanceHintRequestValue(reqs->second));
                        }
                        LOG_DEBUG_TAG("BATCHING:%s:%ld", "user requested size", requests);
                        if (!requests) { // no limitations from user
                            requests = optimalBatchSize * upperBoundStreamsNum * 2;
                            LOG_DEBUG_TAG("BATCHING:%s:%ld", "deduced size:", requests);
                        }
                    } catch (const IE::Exception& iie) {
                        LOG_WARNING_TAG("deduce optimal infer requset num for auto-batch failed :%s",
                            iie.what());
                    }
                    real = (std::max)(requests, optimalBatchSize);
                } else if (deviceInfo.deviceName.find("VPUX") != std::string::npos) {
                    real = 8u;
                } else {
                    real = upperBoundStreamsNum ? 2 * upperBoundStreamsNum : defaultNumForTPUT;
                }
            } else {
                real = defaultNumForLatency;
            }
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type {real};
    } else if (name == ov::execution_devices) {
        std::lock_guard<std::mutex> lock(_autoSContext->_confMutex);
        std::vector<std::string> exeDevices = {};
        for (int i = 0; i < CONTEXTNUM; i++) {
            if (_autoSchedule->_loadContext[i].isEnabled && _autoSchedule->_loadContext[i].isAlready) {
                if (i == 0 && !_autoSchedule->_loadContext[CPU].executableNetwork._ptr) {
                    continue;
                } else {
                    exeDevices.push_back(_autoSchedule->_loadContext[i].workName.substr(_autoSchedule->_loadContext[i].workName.find(":") + 1));
                    break;
                }
            }
        }
        return decltype(ov::available_devices)::value_type {exeDevices};
    }
    IE_THROW(NotFound) << name << " not found in the ExecutableNetwork config";
}
}  // namespace MultiDevicePlugin
