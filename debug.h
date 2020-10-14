#ifndef VULKAN_TUTOR_DEBUG_H
#define VULKAN_TUTOR_DEBUG_H

#include <vulkan/vulkan.hpp>

#include "utils.h"

namespace debug {
    using DynamicUniqueDebugUtilsMessengerEXT = vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>;

    namespace internal {
        void checkValidationLayerSupport();
        DynamicUniqueDebugUtilsMessengerEXT setupDebugMessenger(vk::Instance instance,
                                                                const vk::DispatchLoaderDynamic &d);

        VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData);

        constexpr std::array<const char *, 1> validationLayers {"VK_LAYER_KHRONOS_validation"};
    }

    template<bool enableValidationLayers = ::enableValidationLayers>
    inline void checkValidationLayerSupport() {
        if (enableValidationLayers) {
            internal::checkValidationLayerSupport();
        }
    }

    template<bool enableValidationLayers = ::enableValidationLayers>
    constexpr size_t validationLayersSize() {
        return enableValidationLayers ? internal::validationLayers.size() : 0;
    }

    template<bool enableValidationLayers = ::enableValidationLayers>
    constexpr const char * const * validationLayersData() {
        return enableValidationLayers ? internal::validationLayers.data() : nullptr;
    }

    constexpr vk::DebugUtilsMessengerCreateInfoEXT createDebugInfo(
            {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            internal::debugCallback,
            nullptr
    );

    template<bool enableDebugMessenger = ::enableDebugMessenger>
    inline DynamicUniqueDebugUtilsMessengerEXT setupDebugMessenger(vk::Instance instance,
                                                                   const vk::DispatchLoaderDynamic &d) {
        if (!enableDebugMessenger) {
            return {};
        }

        return internal::setupDebugMessenger(instance, d);
    }

    template<bool enableDebugMessenger = ::enableDebugMessenger>
    constexpr void appendDebugInfo(vk::InstanceCreateInfo &createInfo) {
        if (enableDebugMessenger) {
            createInfo.pNext = &debug::createDebugInfo;
        }
    }

    template<bool enableDebugMessenger = ::enableDebugMessenger>
    inline void appendDebugExtension(std::vector<const char *> &extensions) {
        if (enableDebugMessenger) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }
}

#endif //VULKAN_TUTOR_DEBUG_H
