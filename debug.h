#ifndef VULKAN_TUTOR_DEBUG_H
#define VULKAN_TUTOR_DEBUG_H

#include <array>

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

    inline void checkValidationLayerSupport() {
        if (enableValidationLayers) {
            internal::checkValidationLayerSupport();
        }
    }

    constexpr size_t validationLayersSize() {
        return enableValidationLayers ? internal::validationLayers.size() : 0;
    }

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

    inline DynamicUniqueDebugUtilsMessengerEXT setupDebugMessenger(vk::Instance instance,
                                                                   const vk::DispatchLoaderDynamic &d) {
        if (!enableDebugMessenger) {
            return {};
        }

        return internal::setupDebugMessenger(instance, d);
    }

    constexpr void appendDebugInfo(vk::InstanceCreateInfo &createInfo) {
        if (enableDebugMessenger) {
            createInfo.pNext = &debug::createDebugInfo;
        }
    }
}

#endif //VULKAN_TUTOR_DEBUG_H
