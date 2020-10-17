#ifndef VULKAN_TUTOR_COMMON_H
#define VULKAN_TUTOR_COMMON_H

#include <optional>

#include "GLFWInclude.h"

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
