#ifndef VULKAN_TUTOR_COMMON_H
#define VULKAN_TUTOR_COMMON_H

#include <optional>

#include "GLFWInclude.h"

using OptRefUniqueFence = std::optional<std::reference_wrapper<vk::UniqueFence>>;

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily{};
    std::optional<uint32_t> presentFamily{};
    std::optional<uint32_t> transferFamily{};

    [[nodiscard]] constexpr bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities{};
    std::vector<vk::SurfaceFormatKHR> formats{};
    std::vector<vk::PresentModeKHR> presentModes{};
};

struct SwapChain {
    vk::UniqueSwapchainKHR swapChain{};
    vk::Format imageFormat{};
    vk::Extent2D extent{};
};

struct BufferWithMemory {
    vk::UniqueDeviceMemory bufferMemory{};
    vk::UniqueBuffer buffer{};
};

struct ImageWithMemory {
    vk::UniqueDeviceMemory imageMemory{};
    vk::UniqueImage image{};
};

struct SwitchLayout {
    vk::ImageLayout oldLayout{};
    vk::ImageLayout newLayout{};

    bool operator ==(const SwitchLayout &other) const {
        return oldLayout == other.oldLayout && newLayout == other.newLayout;
    };
};

#endif //VULKAN_TUTOR_COMMON_H
