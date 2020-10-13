#include "debug.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <iostream>

#include <vulkan/vulkan.hpp>

#include "utils.h"

namespace debug::internal {
    namespace {
        constexpr vk::DebugUtilsMessageTypeFlagsEXT allowedMessageTypes =
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;

        constexpr bool filterLog(const vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                 const vk::DebugUtilsMessageTypeFlagsEXT messageType) {
            return messageSeverity != vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                || (messageType & allowedMessageTypes);
        }

        void debugCallback(
                const vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                const vk::DebugUtilsMessageTypeFlagsEXT messageType,
                const vk::DebugUtilsMessengerCallbackDataEXT &callbackData) {
            if (filterLog(messageSeverity, messageType)) {
                std::cerr << print_time << vk::to_string(messageSeverity) << ' '
                          << vk::to_string(messageType)
                          << ' ' << callbackData.pMessage << std::endl;
            }
        }
    }

    void checkValidationLayerSupport() {
        const auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const auto validationLayer : validationLayers) {
            if (std::none_of(
                    availableLayers.cbegin(), availableLayers.cend(),
                    [&validationLayer](const auto &availableLayer) {
                        return std::strcmp(validationLayer, availableLayer.layerName) == 0;
                    })) {
                throw vk::LayerNotPresentError(validationLayer);
            }
        }
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
            void */*pUserData*/) {
        debugCallback(
                static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity),
                static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageType),
                *pCallbackData
        );
        return VK_FALSE;
    }

    DynamicUniqueDebugUtilsMessengerEXT setupDebugMessenger(vk::Instance instance,
                                                            const vk::DispatchLoaderDynamic &d) {
        return instance.createDebugUtilsMessengerEXTUnique<vk::DispatchLoaderDynamic>(
                debug::createDebugInfo, nullptr, d
        );
    }
}
