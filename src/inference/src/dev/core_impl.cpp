// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core_impl.hpp"

#include <memory>

#include "any_copy.hpp"
#include "check_network_batchable.hpp"
#include "compilation_context.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "dev/converter_utils.hpp"
#include "file_utils.h"
#include "ie_itt.hpp"
#include "ie_network_reader.hpp"
#include "ie_ngraph_utils.hpp"
#include "iplugin_wrapper.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/version.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "preprocessing/preprocessing.hpp"
#include "xml_parse_utils.h"

ov::ICore::~ICore() = default;

namespace {

template <typename F>
void allowNotImplemented(F&& f) {
    try {
        f();
    } catch (const InferenceEngine::NotImplemented&) {
    } catch (const ov::NotImplemented&) {
    }
}

void stripDeviceName(std::string& device, const std::string& substr) {
    auto pos = device.find(substr);
    if (pos == 0) {
        device.erase(pos, substr.length());
    }
}

}  // namespace

ov::CoreImpl::CoreImpl(bool _newAPI) : m_new_api(_newAPI) {
    add_mutex("");  // Register global mutex
    m_executor_manager = ov::threading::executor_manager();
    for (const auto& it : ov::get_available_opsets()) {
        opsetNames.insert(it.first);
    }
}

void ov::CoreImpl::register_plugins_in_registry(const std::string& xml_config_file, const bool& by_abs_path) {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto parse_result = ParseXml(xml_config_file.c_str());
    if (!parse_result.error_msg.empty()) {
        IE_THROW() << parse_result.error_msg;
    }

    pugi::xml_document& xmlDoc = *parse_result.xml;

    using namespace XMLParseUtils;
    pugi::xml_node ieNode = xmlDoc.document_element();
    pugi::xml_node devicesNode = ieNode.child("plugins");

    FOREACH_CHILD (pluginNode, devicesNode, "plugin") {
        std::string deviceName = GetStrAttr(pluginNode, "name");
        if (pluginRegistry.find(deviceName) != pluginRegistry.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\"  is already registered in the OpenVINO Runtime";
        }
        if (deviceName.find('.') != std::string::npos) {
            IE_THROW() << "Device name must not contain dot '.' symbol";
        }

        ov::util::FilePath pluginPath =
            ov::util::get_plugin_path(GetStrAttr(pluginNode, "location"), xml_config_file, by_abs_path);

        // check properties
        auto propertiesNode = pluginNode.child("properties");
        ov::AnyMap config;

        if (propertiesNode) {
            FOREACH_CHILD (propertyNode, propertiesNode, "property") {
                std::string key = GetStrAttr(propertyNode, "key");
                std::string value = GetStrAttr(propertyNode, "value");
                config[key] = value;
            }
        }

        // check extensions
        auto extensionsNode = pluginNode.child("extensions");
        std::vector<ov::util::FilePath> listOfExtentions;

        if (extensionsNode) {
            FOREACH_CHILD (extensionNode, extensionsNode, "extension") {
                ov::util::FilePath extensionLocation =
                    ov::util::to_file_path(GetStrAttr(extensionNode, "location").c_str());
                listOfExtentions.push_back(extensionLocation);
            }
        }

        // fill value in plugin registry for later lazy initialization
        {
            PluginDescriptor desc{pluginPath, config, listOfExtentions};
            pluginRegistry[deviceName] = desc;
            add_mutex(deviceName);
        }
    }
}

ov::Plugin ov::CoreImpl::get_plugin(const std::string& pluginName) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "CoreImpl::get_plugin");

    auto deviceName = pluginName;
    if (deviceName == ov::DEFAULT_DEVICE_NAME)
        deviceName = "AUTO";
    stripDeviceName(deviceName, "-");
    std::map<std::string, PluginDescriptor>::const_iterator it;
    {
        // Global lock to find plugin.
        // Always use global mutex if iterate over plugins or pluginRegistry
        std::lock_guard<std::mutex> g_lock(get_mutex());

        // Plugin is not created, check that plugin is registered
        it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
            if (pluginName == ov::DEFAULT_DEVICE_NAME)
                IE_THROW() << "No device is provided, so AUTO device is used by default, which failed loading.";
            else
                IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the OpenVINO Runtime";
        }
    }
    std::lock_guard<std::mutex> lock(get_mutex(deviceName));

    PluginDescriptor desc;
    {
        // Global lock to find plugin.
        // Always use global mutex if iterate over plugins or pluginRegistry
        std::lock_guard<std::mutex> g_lock(get_mutex());
        auto it_plugin = plugins.find(deviceName);
        if (it_plugin != plugins.end())
            return it_plugin->second;

        desc = it->second;
    }
    // Plugin is in registry, but not created, let's create
    std::shared_ptr<void> so;
    try {
        ov::Plugin plugin;

        if (desc.pluginCreateFunc) {  // static OpenVINO case
            std::shared_ptr<ov::IPlugin> plugin_impl;
            desc.pluginCreateFunc(plugin_impl);
            plugin = Plugin{plugin_impl, {}};
        } else {
            so = ov::util::load_shared_object(desc.libraryLocation.c_str());
            std::shared_ptr<ov::IPlugin> plugin_impl;
            reinterpret_cast<InferenceEngine::CreatePluginEngineFunc*>(
                ov::util::get_symbol(so, InferenceEngine::create_plugin_function))(plugin_impl);
            plugin = Plugin{plugin_impl, so};
        }

        {
            plugin.set_name(deviceName);

            // Set Core class reference to plugins
            std::weak_ptr<InferenceEngine::ICore> mutableCore =
                std::const_pointer_cast<InferenceEngine::ICore>(shared_from_this());
            plugin.set_core(mutableCore);
        }

        // Add registered extensions to new plugin
        allowNotImplemented([&]() {
            for (const auto& ext : extensions) {
                plugin.add_extension(ext);
            }
        });

        // configuring
        {
            if (device_supports_cache_dir(plugin)) {
                auto cacheConfig = coreConfig.get_cache_config_for_device(deviceName);
                if (cacheConfig._cacheManager) {
                    desc.defaultConfig[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                }
            } else if (desc.defaultConfig.count(CONFIG_KEY(CACHE_DIR)) > 0) {
                // Remove "CACHE_DIR" from config if it is not supported by plugin
                desc.defaultConfig.erase(CONFIG_KEY(CACHE_DIR));
            }
            allowNotImplemented([&]() {
                // Add device specific value to support device_name.device_id cases
                auto supportedConfigKeys =
                    plugin.get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {}).as<std::vector<std::string>>();
                auto config_iter = std::find(supportedConfigKeys.begin(),
                                             supportedConfigKeys.end(),
                                             CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID));
                const bool supportsConfigDeviceID = config_iter != supportedConfigKeys.end();
                const std::string deviceKey =
                    supportsConfigDeviceID ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID) : CONFIG_KEY(DEVICE_ID);

                for (auto pluginDesc : pluginRegistry) {
                    InferenceEngine::DeviceIDParser parser(pluginDesc.first);
                    if (pluginDesc.first.find(deviceName) != std::string::npos && !parser.getDeviceID().empty()) {
                        pluginDesc.second.defaultConfig[deviceKey] = parser.getDeviceID();
                        plugin.set_property(pluginDesc.second.defaultConfig);
                    }
                }
                plugin.set_property(desc.defaultConfig);
            });

            allowNotImplemented([&]() {
                for (auto&& extensionLocation : desc.listOfExtentions) {
                    plugin.add_extension(std::make_shared<InferenceEngine::Extension>(extensionLocation));
                }
            });
        }

        std::lock_guard<std::mutex> g_lock(get_mutex());
        // add plugin as extension itself
        if (desc.extensionCreateFunc) {  // static OpenVINO case
            try {
                InferenceEngine::IExtensionPtr ext;
                desc.extensionCreateFunc(ext);
                AddExtensionUnsafe(ext);
            } catch (const InferenceEngine::GeneralError&) {
                // the same extension can be registered multiple times - ignore it!
            }
        } else {
            TryToRegisterLibraryAsExtensionUnsafe(desc.libraryLocation);
        }

        return plugins.emplace(deviceName, plugin).first->second;
    } catch (const InferenceEngine::Exception& ex) {
        IE_THROW() << "Failed to create plugin " << ov::util::from_file_path(desc.libraryLocation) << " for device "
                   << deviceName << "\n"
                   << "Please, check your environment\n"
                   << ex.what() << "\n";
    }
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::compile_model::model");
    std::string deviceName = device_name;
    ov::AnyMap config_with_batch = config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    apply_auto_batching(model, deviceName, config_with_batch);
    clean_properties(deviceName, config_with_batch, ov::auto_batch_timeout);

    bool forceDisableCache = config_with_batch.count(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE)) > 0;
    auto parsed = parseDeviceNameIntoConfig(deviceName, config_with_batch);
    if (forceDisableCache) {
        // remove this config key from parsed as plugins can throw unsupported exception
        parsed._config.erase(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE));
    }
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> res;
    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (!forceDisableCache && cacheManager && device_supports_import_export(plugin)) {
        cacheContent.blobId = ov::NetworkCompilationContext::compute_hash(
            model,
            create_compile_config(plugin, parsed._deviceName, parsed._config));
        bool loadedFromCache = false;
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, parsed._config, {}, loadedFromCache);
        if (!loadedFromCache) {
            res = compile_model_impl(model, plugin, parsed._config, {}, cacheContent, forceDisableCache);
        }
    } else {
        res = compile_model_impl(model, plugin, parsed._config, {}, cacheContent, forceDisableCache);
    }
    return {res._ptr, res._so};
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::RemoteContext& context,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::compile_model::RemoteContext");
    if (context._impl == nullptr) {
        IE_THROW() << "Remote context is null";
    }
    // have to deduce the device name/config from the context first
    auto parsed = parseDeviceNameIntoConfig(context.get_device_name(), config);
    std::string& deviceName = parsed._deviceName;
    auto& config_with_batch = parsed._config;
    // if auto-batching is applicable, the below function will patch the device name and config accordingly:
    apply_auto_batching(model, deviceName, config_with_batch);
    clean_properties(deviceName, config_with_batch, ov::auto_batch_timeout);
    parsed = parseDeviceNameIntoConfig(deviceName, config_with_batch);

    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> res;
    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (cacheManager && device_supports_import_export(plugin)) {
        cacheContent.blobId = ov::NetworkCompilationContext::compute_hash(
            model,
            create_compile_config(plugin, parsed._deviceName, parsed._config));
        bool loadedFromCache = false;
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, parsed._config, context, loadedFromCache);
        if (!loadedFromCache) {
            res = compile_model_impl(model, plugin, parsed._config, context, cacheContent);
        }
    } else {
        res = compile_model_impl(model, plugin, parsed._config, context, cacheContent);
    }
    return res;
}
ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(ov::Plugin& plugin,
                                                          const std::shared_ptr<const ov::Model>& model,
                                                          const ov::RemoteContext& context,
                                                          const ov::AnyMap& config) const {
    std::shared_ptr<const ov::Model> prepared_model = model;
    ov::SoPtr<ov::ICompiledModel> compiled_model;

    if (!is_new_api() && !std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(plugin.m_ptr)) {
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::AddPreprocessing>();

        auto cloned_model = model->clone();
        manager.run_passes(cloned_model);
        prepared_model = cloned_model;
    }

    if (!context._impl) {
        compiled_model = plugin.compile_model(prepared_model, config);
    } else {
        compiled_model = plugin.compile_model(prepared_model, context, config);
    }
    return compiled_model;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::string& model_path,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::compile_model::Path");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> res;
    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager, model_path};
    if (cacheManager && device_supports_import_export(plugin)) {
        bool loadedFromCache = false;
        cacheContent.blobId = ov::NetworkCompilationContext::compute_hash(
            model_path,
            create_compile_config(plugin, parsed._deviceName, parsed._config));
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, parsed._config, {}, loadedFromCache);
        if (!loadedFromCache) {
            auto cnnNetwork = ReadNetwork(model_path, std::string());
            res = compile_model_impl(cnnNetwork.getFunction(), plugin, parsed._config, {}, cacheContent);
        }
    } else if (cacheManager) {
        res = plugin.compile_model(model_path, parsed._config);
    } else {
        auto cnnNetwork = ReadNetwork(model_path, std::string());
        res = compile_model_impl(cnnNetwork.getFunction(), plugin, parsed._config, {}, cacheContent);
    }
    return {res._ptr, res._so};
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model(const std::string& model_str,
                                                          const ov::Tensor& weights,
                                                          const std::string& device_name,
                                                          const ov::AnyMap& config) const {
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    auto plugin = get_plugin(parsed._deviceName);
    ov::SoPtr<ov::ICompiledModel> res;

    auto cacheManager =
        coreConfig.get_cache_config_for_device(parsed._deviceName, device_supports_cache_dir(plugin), parsed._config)
            ._cacheManager;
    auto cacheContent = CacheContent{cacheManager};
    if (cacheManager && device_supports_import_export(plugin)) {
        bool loadedFromCache = false;
        cacheContent.blobId = ov::NetworkCompilationContext::compute_hash(
            model_str,
            weights,
            create_compile_config(plugin, parsed._deviceName, parsed._config));
        auto lock = cacheGuard.get_hash_lock(cacheContent.blobId);
        res = load_model_from_cache(cacheContent, plugin, parsed._config, {}, loadedFromCache);
        if (!loadedFromCache) {
            auto cnnNetwork = read_model(model_str, weights);
            res = compile_model_impl(cnnNetwork, plugin, parsed._config, {}, cacheContent);
        }
    } else {
        auto cnnNetwork = read_model(model_str, weights);
        res = compile_model_impl(cnnNetwork, plugin, parsed._config, {}, cacheContent);
    }
    return {res._ptr, res._so};
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::import_model(std::istream& model,
                                                         const std::string& device_name,
                                                         const ov::AnyMap& config) const {
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    auto exec = get_plugin(parsed._deviceName).import_model(model, config);

    return {exec._ptr, exec._so};
}

ov::SupportedOpsMap ov::CoreImpl::query_model(const std::shared_ptr<const ov::Model>& model,
                                              const std::string& device_name,
                                              const ov::AnyMap& config) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::query_model");
    auto parsed = parseDeviceNameIntoConfig(device_name, config);
    auto ret = get_plugin(parsed._deviceName).query_model(model, parsed._config);
    return ret;
}

std::vector<std::string> ov::CoreImpl::get_available_devices() const {
    std::vector<std::string> devices;
    const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : get_registered_devices()) {
        std::vector<std::string> devicesIDs;
        try {
            const ie::Parameter p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string>>();
        } catch (const ie::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const ov::Exception&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::runtime_error&) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
            IE_THROW() << "An exception is thrown while trying to create the " << deviceName
                       << " device and call GetMetric: " << ex.what();
        } catch (...) {
            IE_THROW() << "Unknown exception is thrown while trying to create the " << deviceName
                       << " device and call GetMetric";
        }

        if (devicesIDs.size() > 1) {
            for (auto&& deviceID : devicesIDs) {
                devices.push_back(deviceName + '.' + deviceID);
            }
        } else if (!devicesIDs.empty()) {
            devices.push_back(deviceName);
        }
    }

    return devices;
}

ov::RemoteContext ov::CoreImpl::create_context(const std::string& device_name, const AnyMap& args) const {
    auto parsed = ov::parseDeviceNameIntoConfig(device_name, args);
    return get_plugin(parsed._deviceName).create_context(parsed._config);
}

ov::AnyMap ov::CoreImpl::get_supported_property(const std::string& device_name, const ov::AnyMap& config) const {
    std::vector<std::string> supportedConfigKeys;
    try {
        supportedConfigKeys = GetMetric(device_name, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    } catch (ov::Exception&) {
    }
    try {
        for (auto&& property : ICore::get_property(device_name, ov::supported_properties)) {
            if (property.is_mutable()) {
                supportedConfigKeys.emplace_back(std::move(property));
            }
        }
    } catch (ov::Exception&) {
    }
    ov::AnyMap supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    for (auto&& config : config) {
        auto parsed = parseDeviceNameIntoConfig(config.first);
        if (device_name.find(parsed._deviceName) != std::string::npos) {
            std::stringstream strm(config.second.as<std::string>());
            std::map<std::string, std::string> device_configs;
            util::Read<std::map<std::string, std::string>>{}(strm, device_configs);
            for (auto&& device_config : device_configs) {
                if (util::contains(supportedConfigKeys, device_config.first)) {
                    supportedConfig[device_config.first] = device_config.second;
                }
            }
            for (auto&& config : parsed._config) {
                supportedConfig[config.first] = config.second.as<std::string>();
            }
        }
    }
    return supportedConfig;
}

bool ov::CoreImpl::is_new_api() const {
    return m_new_api;
}

ov::RemoteContext ov::CoreImpl::get_default_context(const std::string& device_name) const {
    auto parsed = ov::parseDeviceNameIntoConfig(device_name, ov::AnyMap{});
    return get_plugin(parsed._deviceName).get_default_context(parsed._config);
}

void ov::CoreImpl::apply_auto_batching(const std::shared_ptr<const ov::Model>& model,
                                       std::string& deviceName,
                                       ov::AnyMap& config) const {
    std::string deviceNameWithBatchSize, deviceNameWithoutBatch;
    // fully strict dims tracking by default (Auto-Batching is enabled implicitly)
    bool strictly_check_dims = true;
    if (deviceName.find("BATCH") != std::string::npos) {
        // explicitly enabled Auto-Batching
        auto pos = deviceName.find_first_of(":");
        if (pos == std::string::npos)
            return;  // BATCH device is already configured via the config
        deviceNameWithBatchSize = deviceName.substr(pos + 1);
        deviceNameWithoutBatch = InferenceEngine::DeviceIDParser::getBatchDevice(deviceNameWithBatchSize);
        // when user sets the BATCH device explicitly, we may check the dims less strictly
        // as the result is being checked by the user
        strictly_check_dims = false;
    } else {
        // check if Auto-Batch plugin registered
        try {
            get_plugin("BATCH");
        } catch (const std::runtime_error&) {
            return;
        }
        // check whether the Auto-Batching is disabled explicitly
        const auto& batch_mode = config.find(ov::hint::allow_auto_batching.name());
        if (batch_mode != config.end()) {
            const auto disabled = batch_mode->second.as<std::string>() == CONFIG_VALUE(NO);
            // virtual plugins like AUTO/MULTI will need the config
            // e.g to deduce the #requests correctly
            // otherwise, no need for this config key in the rest of loading
            if (deviceName.find("AUTO") == std::string::npos && deviceName.find("MULTI") == std::string::npos)
                config.erase(batch_mode);
            if (disabled)
                return;
        } else if (!coreConfig.flag_allow_auto_batching) {
            return;
        }
        // check whether if the Auto-Batching is applicable to the device
        auto device = ov::parseDeviceNameIntoConfig(deviceName);
        deviceNameWithoutBatch = deviceName;
        auto d = device._deviceName;
        std::vector<std::string> metrics =
            get_plugin(d).get_property(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
        auto it = std::find(metrics.begin(), metrics.end(), METRIC_KEY(OPTIMAL_BATCH_SIZE));
        if (metrics.end() == it)
            return;
        // if applicable, the Auto-Batching is implicitly enabled via the performance hints
        bool bTputInPlg = GetConfig(d, CONFIG_KEY(PERFORMANCE_HINT)).as<std::string>() == CONFIG_VALUE(THROUGHPUT);
        const auto& mode = config.find(CONFIG_KEY(PERFORMANCE_HINT));
        bool bTputInLoadCfg = (mode != config.end() && mode->second.as<std::string>() == CONFIG_VALUE(THROUGHPUT));
        const auto& excl = config.find(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));
        bool bExclReqsEnabled = (excl != config.end() && excl->second.as<std::string>() == CONFIG_VALUE(YES));
        if (bExclReqsEnabled || (!bTputInPlg && !bTputInLoadCfg))
            return;
    }
    auto batchConfig = deviceNameWithBatchSize.empty() ? deviceNameWithoutBatch : deviceNameWithBatchSize;
    auto res = ov::details::is_model_batchable(model, deviceNameWithoutBatch, strictly_check_dims);
    switch (res) {
    case ov::details::NetworkBatchAbility::NO:
        return;
    case ov::details::NetworkBatchAbility::AS_IS:
        deviceName = "BATCH:" + batchConfig;
        break;
    case ov::details::NetworkBatchAbility::WITH_HETERO:
        deviceName = "HETERO:BATCH," + deviceNameWithoutBatch;
        config[CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG)] = batchConfig;
        break;
    }
}

void ov::CoreImpl::clean_properties(std::string& deviceName, ov::AnyMap& config, ov::Any property) const {
    // auto-batching is not applicable, if there is auto_batch_timeout, delete it
    if (deviceName.find("BATCH") == std::string::npos) {
        const auto& batch_timeout_mode = config.find(property.as<std::string>());
        if (batch_timeout_mode != config.end()) {
            if (deviceName.find("AUTO") == std::string::npos && deviceName.find("MULTI") == std::string::npos)
                config.erase(batch_timeout_mode);
        }
    }
}

void ov::CoreImpl::set_property(const std::string& device_name, const AnyMap& properties) {
    OPENVINO_ASSERT(device_name.find("HETERO:") != 0,
                    "set_property is supported only for HETERO itself (without devices). "
                    "You can configure the devices with set_property before creating the HETERO on top.");
    OPENVINO_ASSERT(device_name.find("MULTI:") != 0,
                    "set_property is supported only for MULTI itself (without devices). "
                    "You can configure the devices with set_property before creating the MULTI on top.");
    OPENVINO_ASSERT(device_name.find("AUTO:") != 0,
                    "set_property is supported only for AUTO itself (without devices). "
                    "You can configure the devices with set_property before creating the AUTO on top.");
    OPENVINO_ASSERT(device_name.find("BATCH:") != 0,
                    "set_property is supported only for BATCH itself (without devices). "
                    "You can configure the devices with set_property before creating the BATCH on top.");

    // unsupport to set ov::device::properties to HW device through this function
    auto devices = get_registered_devices();
    for (auto&& config : properties) {
        auto parsed = parseDeviceNameIntoConfig(config.first);
        auto is_secondary_config_for_hw_device =
            std::any_of(devices.begin(), devices.end(), [&](const std::string& device) {
                return device == parsed._deviceName;
            });
        OPENVINO_ASSERT(!is_secondary_config_for_hw_device,
                        "set_property do not support ov::device::propreties. "
                        "You can configure the devices through the compile_model()/loadNetwork() API.");
    }
    set_property_for_device(properties, device_name);
}

ov::Any ov::CoreImpl::get_property_for_core(const std::string& name) const {
    if (name == ov::force_tbb_terminate.name()) {
        const auto flag = ov::threading::executor_manager()->get_property(name).as<bool>();
        return decltype(ov::force_tbb_terminate)::value_type(flag);
    } else if (name == ov::cache_dir.name()) {
        return ov::Any(coreConfig.get_cache_dir());
    } else if (name == ov::hint::allow_auto_batching.name()) {
        const auto flag = coreConfig.flag_allow_auto_batching;
        return decltype(ov::hint::allow_auto_batching)::value_type(flag);
    }

    OPENVINO_UNREACHABLE("Exception is thrown while trying to call get_property with unsupported property: '",
                         name,
                         "'");
}

ov::Any ov::CoreImpl::get_property(const std::string& device_name,
                                   const std::string& name,
                                   const AnyMap& arguments) const {
    OPENVINO_ASSERT(device_name.find("HETERO:") != 0,
                    "You can only get_property of the HETERO itself (without devices). "
                    "get_property is also possible for the individual devices before creating the HETERO on top.");
    OPENVINO_ASSERT(device_name.find("MULTI:") != 0,
                    "You can only get_property of the MULTI itself (without devices). "
                    "get_property is also possible for the individual devices before creating the MULTI on top.");
    OPENVINO_ASSERT(device_name.find("AUTO:") != 0,
                    "You can only get_property of the AUTO itself (without devices). "
                    "get_property is also possible for the individual devices before creating the AUTO on top.");
    OPENVINO_ASSERT(device_name.find("BATCH:") != 0,
                    "You can only get_property of the BATCH itself (without devices). "
                    "get_property is also possible for the individual devices before creating the BATCH on top.");

    if (device_name.empty()) {
        return get_property_for_core(name);
    }

    auto parsed = parseDeviceNameIntoConfig(device_name, arguments);
    return get_plugin(parsed._deviceName).get_property(name, parsed._config);
}

void ov::CoreImpl::unload_plugin(const std::string& deviceName) {
    std::lock_guard<std::mutex> lock(get_mutex());
    auto it = plugins.find(deviceName);
    if (it == plugins.end()) {
        IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the OpenVINO Runtime";
    }

    plugins.erase(deviceName);
}

void ov::CoreImpl::register_plugin(const std::string& plugin, const std::string& device_name) {
    std::lock_guard<std::mutex> lock(get_mutex());

    auto it = pluginRegistry.find(device_name);
    if (it != pluginRegistry.end()) {
        IE_THROW() << "Device with \"" << device_name << "\"  is already registered in the OpenVINO Runtime";
    }

    if (device_name.find('.') != std::string::npos) {
        IE_THROW() << "Device name must not contain dot '.' symbol";
    }

    PluginDescriptor desc{ov::util::get_plugin_path(plugin)};
    pluginRegistry[device_name] = desc;
    add_mutex(device_name);
}

/**
 * @brief Provides a list of plugin names in registry; physically such plugins may not be created
 * @return A list of plugin names
 */
std::vector<std::string> ov::CoreImpl::get_registered_devices() const {
    std::lock_guard<std::mutex> lock(get_mutex());

    std::vector<std::string> listOfDevices;
    for (auto&& pluginDesc : pluginRegistry) {
        listOfDevices.push_back(pluginDesc.first);
    }

    return listOfDevices;
}

/**
 * @brief Sets config values for a plugin or set of plugins
 * @param deviceName A device name to set config to
 *        If empty, config is set for all the plugins / plugin's meta-data
 * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
 *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
 */
void ov::CoreImpl::set_property_for_device(const ov::AnyMap& configMap, const std::string& deviceName) {
    auto config = configMap;
    if (config.empty()) {
        return;
    }

    InferenceEngine::DeviceIDParser parser(deviceName);
    std::string clearDeviceName = parser.getDeviceName();

    std::vector<std::pair<std::string, ov::Plugin>> created_plugins;
    {
        std::lock_guard<std::mutex> lock(get_mutex());
        created_plugins.reserve(plugins.size());

        if (deviceName.empty()) {
            coreConfig.set_and_update(config);
        } else {
            auto cache_it = config.find(CONFIG_KEY(CACHE_DIR));
            if (cache_it != config.end()) {
                coreConfig.set_cache_dir_for_device((cache_it->second).as<std::string>(), clearDeviceName);
            }
        }

        auto base_desc = pluginRegistry.find(clearDeviceName);
        if (pluginRegistry.find(deviceName) == pluginRegistry.end() && base_desc != pluginRegistry.end()) {
            PluginDescriptor desc{base_desc->second.libraryLocation, config, base_desc->second.listOfExtentions};
            pluginRegistry[deviceName] = desc;
        }

        // set config for plugins in registry
        bool configIsSet = false;
        for (auto& desc : pluginRegistry) {
            if (deviceName.empty() || deviceName == desc.first) {
                for (auto&& conf : config) {
                    desc.second.defaultConfig[conf.first] = conf.second;
                }
                configIsSet = true;
            }
        }

        if (!configIsSet && !deviceName.empty()) {
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the OpenVINO Runtime";
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
            if (deviceName.empty() || clearDeviceName == plugin.first) {
                created_plugins.emplace_back(std::pair<std::string, ov::Plugin>{plugin.first, plugin.second});
            }
        }
    }
    for (auto& plugin : created_plugins) {
        allowNotImplemented([&]() {
            std::lock_guard<std::mutex> lock(get_mutex(plugin.first));
            auto configCopy = config;
            if (device_supports_cache_dir(plugin.second)) {
                auto cacheConfig = coreConfig.get_cache_config_for_device(deviceName);
                if (cacheConfig._cacheManager) {
                    configCopy[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                }
            } else if (configCopy.count(CONFIG_KEY(CACHE_DIR)) > 0) {
                // Remove "CACHE_DIR" from config if it is not supported by plugin
                configCopy.erase(CONFIG_KEY(CACHE_DIR));
            }
            // Add device specific value to support device_name.device_id cases
            auto supportedConfigKeys =
                plugin.second.get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {}).as<std::vector<std::string>>();
            auto config_iter = std::find(supportedConfigKeys.begin(),
                                         supportedConfigKeys.end(),
                                         CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID));
            const bool supportsConfigDeviceID = config_iter != supportedConfigKeys.end();
            const std::string deviceKey =
                supportsConfigDeviceID ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID) : CONFIG_KEY(DEVICE_ID);

            if (!parser.getDeviceID().empty()) {
                configCopy[deviceKey] = parser.getDeviceID();
            }
            plugin.second.set_property(configCopy);
        });
    }
}

void ov::CoreImpl::add_extension(const std::vector<ov::Extension::Ptr>& extensions) {
    std::lock_guard<std::mutex> lock(get_mutex());
    for (const auto& ext : extensions) {
        ov_extensions.emplace_back(ext);
        if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext)) {
            for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
                ov_extensions.emplace_back(attached_ext);
            }
        }
    }
}

const std::vector<InferenceEngine::IExtensionPtr>& ov::CoreImpl::GetExtensions() const {
    return extensions;
}

bool ov::CoreImpl::device_supports_import_export(const std::string& deviceName) const {
    auto parsed = parseDeviceNameIntoConfig(deviceName);
    auto plugin = get_plugin(parsed._deviceName);
    return device_supports_import_export(plugin);
}

bool ov::CoreImpl::device_supports_property(const ov::Plugin& plugin, const std::string& key) const {
    return util::contains(plugin.get_property(ov::supported_properties), key);
}

bool ov::CoreImpl::device_supports_import_export(const ov::Plugin& plugin) const {
    auto supportedMetricKeys = plugin.get_property(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
    auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), METRIC_KEY(IMPORT_EXPORT_SUPPORT));
    auto supported =
        (it != supportedMetricKeys.end()) && plugin.get_property(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {}).as<bool>();
    if (!supported) {
        if (device_supports_property(plugin, ov::device::capabilities.name())) {
            supported =
                util::contains(plugin.get_property(ov::device::capabilities), ov::device::capability::EXPORT_IMPORT);
        }
    }
    return supported;
}

bool ov::CoreImpl::device_supports_cache_dir(const ov::Plugin& plugin) const {
    return util::contains(plugin.get_property(ov::supported_properties), ov::cache_dir);
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::compile_model_impl(const std::shared_ptr<const ov::Model>& model,
                                                               ov::Plugin& plugin,
                                                               const ov::AnyMap& parsedConfig,
                                                               const ov::RemoteContext& context,
                                                               const CacheContent& cacheContent,
                                                               bool forceDisableCache) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "CoreImpl::compile_model_impl");
    ov::SoPtr<ov::ICompiledModel> execNetwork;
    execNetwork = compile_model(plugin, model, context, parsedConfig);
    if (!forceDisableCache && cacheContent.cacheManager && device_supports_import_export(plugin)) {
        try {
            // need to export network for further import from "cache"
            OV_ITT_SCOPE(FIRST_INFERENCE, InferenceEngine::itt::domains::IE_LT, "Core::compile_model::Export");
            cacheContent.cacheManager->write_cache_entry(cacheContent.blobId, [&](std::ostream& networkStream) {
                networkStream << ov::CompiledBlobHeader(
                    InferenceEngine::GetInferenceEngineVersion()->buildNumber,
                    ov::NetworkCompilationContext::calculate_file_info(cacheContent.modelPath));
                execNetwork->export_model(networkStream);
            });
        } catch (...) {
            cacheContent.cacheManager->remove_cache_entry(cacheContent.blobId);
            throw;
        }
    }
    return execNetwork;
}

ov::SoPtr<ov::ICompiledModel> ov::CoreImpl::load_model_from_cache(const CacheContent& cacheContent,
                                                                  ov::Plugin& plugin,
                                                                  const ov::AnyMap& config,
                                                                  const ov::RemoteContext& context,
                                                                  bool& networkIsImported) {
    ov::SoPtr<ov::ICompiledModel> execNetwork;
    struct HeaderException {};

    OPENVINO_ASSERT(cacheContent.cacheManager != nullptr);
    try {
        cacheContent.cacheManager->read_cache_entry(cacheContent.blobId, [&](std::istream& networkStream) {
            OV_ITT_SCOPE(FIRST_INFERENCE,
                         InferenceEngine::itt::domains::IE_LT,
                         "Core::LoadNetworkFromCache::ReadStreamAndImport");
            try {
                ov::CompiledBlobHeader header;
                networkStream >> header;
                if (header.getIeVersion() != InferenceEngine::GetInferenceEngineVersion()->buildNumber) {
                    // Build number mismatch, don't use this cache
                    throw InferenceEngine::NetworkNotRead("Version does not match");
                }
                if (header.getFileInfo() !=
                    ov::NetworkCompilationContext::calculate_file_info(cacheContent.modelPath)) {
                    // Original file is changed, don't use cache
                    throw InferenceEngine::NetworkNotRead("Original model file is changed");
                }
            } catch (...) {
                throw HeaderException();
            }

            execNetwork = context._impl ? plugin.import_model(networkStream, context, config)
                                        : plugin.import_model(networkStream, config);
            networkIsImported = true;
            execNetwork->loaded_from_cache();
        });
    } catch (const HeaderException&) {
        // For these exceptions just remove old cache and set that import didn't work
        cacheContent.cacheManager->remove_cache_entry(cacheContent.blobId);
        networkIsImported = false;
    } catch (...) {
        cacheContent.cacheManager->remove_cache_entry(cacheContent.blobId);
        networkIsImported = false;
        // TODO: temporary disabled by #54335. In future don't throw only for new 'blob_outdated' exception
        // throw;
    }
    return execNetwork;
}

ov::AnyMap ov::CoreImpl::create_compile_config(const ov::Plugin& plugin,
                                               const std::string& deviceFamily,
                                               const ov::AnyMap& origConfig) const {
    ov::AnyMap getMetricConfig;
    ov::AnyMap compileConfig;

    // 0. Move TARGET_FALLBACK key to getMetricConfig
    auto targetFallbackIt = origConfig.find("TARGET_FALLBACK");
    if (targetFallbackIt == origConfig.end()) {
        targetFallbackIt = origConfig.find(ov::device::priorities.name());
    }
    if (targetFallbackIt != origConfig.end()) {
        getMetricConfig[targetFallbackIt->first] = targetFallbackIt->second.as<std::string>();
    }

    // 1. Move DEVICE_ID key to getMetricConfig
    auto deviceIt = origConfig.find(ov::device::id.name());
    if (deviceIt != origConfig.end()) {
        getMetricConfig[deviceIt->first] = deviceIt->second.as<std::string>();
    }

    // 2. Replace it with DEVICE_ARCHITECTURE value
    if (device_supports_property(plugin, ov::device::architecture.name())) {
        compileConfig[ov::device::architecture.name()] = plugin.get_property(ov::device::architecture, getMetricConfig);
    } else {
        // Take device name if device does not support DEVICE_ARCHITECTURE metric
        compileConfig[ov::device::architecture.name()] = deviceFamily;
    }

    // 3. Extract config keys which affect compile config
    if (device_supports_property(plugin, ov::caching_properties.name())) {
        auto cachingProps = plugin.get_property(ov::caching_properties);
        for (const auto& prop : cachingProps) {
            // origConfig values have higher priority than plugin parameters
            auto it = origConfig.find(prop);
            compileConfig[prop] = it == origConfig.end() ? plugin.get_property(prop, {}) : it->second;
        }
    }
    return compileConfig;
}

void ov::CoreImpl::AddExtensionUnsafe(const InferenceEngine::IExtensionPtr& extension) const {
    std::map<std::string, ngraph::OpSet> opsets = extension->getOpSets();
    for (const auto& it : opsets) {
        if (opsetNames.find(it.first) != opsetNames.end())
            IE_THROW() << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
        opsetNames.insert(it.first);
    }

    // add extensions for already created plugins
    for (auto& plugin : plugins) {
        try {
            plugin.second.add_extension(extension);
        } catch (...) {
        }
    }
    extensions.emplace_back(extension);
}

void ov::CoreImpl::CoreConfig::set_and_update(ov::AnyMap& config) {
    auto it = config.find(CONFIG_KEY(CACHE_DIR));
    if (it != config.end()) {
        std::lock_guard<std::mutex> lock(_cacheConfigMutex);
        fill_config(_cacheConfig, it->second.as<std::string>());
        for (auto& deviceCfg : _cacheConfigPerDevice) {
            fill_config(deviceCfg.second, it->second.as<std::string>());
        }
        config.erase(it);
    }

    it = config.find(ov::force_tbb_terminate.name());
    if (it != config.end()) {
        auto flag = it->second.as<std::string>() == CONFIG_VALUE(YES) ? true : false;
        ov::threading::executor_manager()->set_property({{it->first, flag}});
        config.erase(it);
    }

    it = config.find(ov::hint::allow_auto_batching.name());
    if (it != config.end()) {
        auto flag = it->second.as<bool>();
        flag_allow_auto_batching = flag;
        config.erase(it);
    }
}

void ov::CoreImpl::CoreConfig::set_cache_dir_for_device(const std::string& dir, const std::string& name) {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    fill_config(_cacheConfigPerDevice[name], dir);
}

std::string ov::CoreImpl::CoreConfig::get_cache_dir() const {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    return _cacheConfig._cacheDir;
}

// Creating thread-safe copy of config including shared_ptr to ICacheManager
// Passing empty or not-existing name will return global cache config
ov::CoreImpl::CoreConfig::CacheConfig ov::CoreImpl::CoreConfig::get_cache_config_for_device(
    const std::string& device_name,
    bool device_supports_cache_dir,
    ov::AnyMap& parsedConfig) const {
    if (parsedConfig.count(CONFIG_KEY(CACHE_DIR))) {
        CoreConfig::CacheConfig tempConfig;
        CoreConfig::fill_config(tempConfig, parsedConfig.at(CONFIG_KEY(CACHE_DIR)).as<std::string>());
        if (!device_supports_cache_dir) {
            parsedConfig.erase(CONFIG_KEY(CACHE_DIR));
        }
        return tempConfig;
    } else {
        std::lock_guard<std::mutex> lock(_cacheConfigMutex);
        if (_cacheConfigPerDevice.count(device_name) > 0) {
            return _cacheConfigPerDevice.at(device_name);
        } else {
            return _cacheConfig;
        }
    }
}

ov::CoreImpl::CoreConfig::CacheConfig ov::CoreImpl::CoreConfig::get_cache_config_for_device(
    const std::string& device_name) const {
    std::lock_guard<std::mutex> lock(_cacheConfigMutex);
    if (_cacheConfigPerDevice.count(device_name) > 0) {
        return _cacheConfigPerDevice.at(device_name);
    } else {
        return _cacheConfig;
    }
}

void ov::CoreImpl::CoreConfig::fill_config(CacheConfig& config, const std::string& dir) {
    config._cacheDir = dir;
    if (!dir.empty()) {
        FileUtils::createDirectoryRecursive(dir);
        config._cacheManager = std::make_shared<ov::FileStorageCacheManager>(dir);
    } else {
        config._cacheManager = nullptr;
    }
}
std::mutex& ov::CoreImpl::get_mutex(const std::string& dev_name) const {
    std::lock_guard<std::mutex> lock(global_mutex);
    try {
        return dev_mutexes.at(dev_name);
    } catch (const std::out_of_range&) {
        throw ov::Exception("Cannot get mutex for device: " + dev_name);
    }
}
void ov::CoreImpl::add_mutex(const std::string& dev_name) {
    std::lock_guard<std::mutex> lock(global_mutex);
    dev_mutexes[dev_name];
}

std::tuple<bool, std::string> ov::CoreImpl::CheckStatic(const InferenceEngine::CNNNetwork& network) {
    bool res = true;
    std::stringstream errMsg;
    auto model = network.getFunction();
    if (model) {
        for (const auto& input : model->inputs()) {
            if (input.get_partial_shape().is_dynamic()) {
                errMsg << "{ input:'";
                for (const auto& name : input.get_names()) {
                    errMsg << name << ",";
                }
                if (auto node = input.get_node_shared_ptr()) {
                    errMsg << node->get_friendly_name();
                }
                errMsg << "', shape=" << input.get_partial_shape() << "} ";
                res = false;
            }
        }
    }
    return {res, errMsg.str()};
}

#ifndef OPENVINO_STATIC_LIBRARY

std::string ov::findPluginXML(const std::string& xmlFile) {
    std::string xmlConfigFile_ = xmlFile;
    if (xmlConfigFile_.empty()) {
        const auto ielibraryDir = ie::getInferenceEngineLibraryPath();

        // plugins.xml can be found in either:

        // 1. openvino-X.Y.Z relative to libopenvino.so folder
        std::ostringstream str;
        str << "openvino-" << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH;
        const auto subFolder = ov::util::to_file_path(str.str());

        // register plugins from default openvino-<openvino version>/plugins.xml config
        ov::util::FilePath xmlConfigFileDefault =
            FileUtils::makePath(FileUtils::makePath(ielibraryDir, subFolder), ov::util::to_file_path("plugins.xml"));
        if (FileUtils::fileExist(xmlConfigFileDefault))
            return xmlConfigFile_ = ov::util::from_file_path(xmlConfigFileDefault);

        // 2. in folder with libopenvino.so
        xmlConfigFileDefault = FileUtils::makePath(ielibraryDir, ov::util::to_file_path("plugins.xml"));
        if (FileUtils::fileExist(xmlConfigFileDefault))
            return xmlConfigFile_ = ov::util::from_file_path(xmlConfigFileDefault);

        throw ov::Exception("Failed to find plugins.xml file");
    }
    return xmlConfigFile_;
}

#endif

ov::AnyMap ov::flatten_sub_properties(const std::string& device, const ov::AnyMap& properties) {
    ov::AnyMap result = properties;
    bool isVirtualDev = device.find("AUTO") != std::string::npos || device.find("MULTI") != std::string::npos ||
                        device.find("HETERO") != std::string::npos;
    for (auto item = result.begin(); item != result.end();) {
        auto parsed = parseDeviceNameIntoConfig(item->first);
        if (!item->second.is<ov::AnyMap>()) {
            item++;
            continue;
        }
        if (device == parsed._deviceName) {
            // 1. flatten the scondary property for target device
            for (auto&& sub_property : item->second.as<ov::AnyMap>()) {
                // 1.1 1st level property overides 2nd level property
                if (result.find(sub_property.first) != result.end())
                    continue;
                result[sub_property.first] = sub_property.second;
            }
            item = result.erase(item);
        } else if (isVirtualDev) {
            // 2. keep the secondary property for the other virtual devices
            item++;
        } else {
            // 3. remove the secondary property setting for other hardware device
            item = result.erase(item);
        }
    }
    return result;
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::string& modelPath, const std::string& binPath) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::read_model from file");
    return ReadNetwork(modelPath, binPath).getFunction();
}

std::shared_ptr<ov::Model> ov::CoreImpl::read_model(const std::string& model,
                                                    const ov::Tensor& weights,
                                                    bool frontendMode) const {
    InferenceEngine::Blob::Ptr blob;
    if (weights) {
        blob = weights._impl;
    }
    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::read_model from memory");
    return ReadNetwork(model, blob, frontendMode).getFunction();
}
