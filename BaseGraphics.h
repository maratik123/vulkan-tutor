#ifndef VULKAN_TUTOR_BASEGRAPHICS_H
#define VULKAN_TUTOR_BASEGRAPHICS_H

#include <optional>
#include <chrono>

#include "GLFWInclude.h"

#include "Common.h"
#include "GLFWWindow.h"
#include "Debug.h"
#include "SizeDependentResources.h"

class BaseGraphics {
public:
    BaseGraphics();
    ~BaseGraphics();
    void run();

private:
    static constexpr int maxFramesInFlight = 2;

    [[nodiscard]] PFN_vkGetInstanceProcAddr getVkGetInstanceProcAddr() const;
    [[nodiscard]] vk::PhysicalDevice pickPhysicalDevice() const;
    [[nodiscard]] vk::UniqueDevice createLogicalDevice() const;
    [[nodiscard]] bool isDeviceSuitable(vk::PhysicalDevice requestedDevice,
                                        vk::PhysicalDeviceType desiredDeviceType) const;
    [[nodiscard]] QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice requestedDevice) const;
    [[nodiscard]] QueueFamilyIndices findQueueFamilies() const { return findQueueFamilies(physicalDevice); }
    [[nodiscard]] SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice requestedDevice) const;
    [[nodiscard]] SwapChainSupportDetails querySwapChainSupport() const {
        return querySwapChainSupport(physicalDevice);
    }
    [[nodiscard]] vk::UniqueCommandPool createCommandPool(std::optional<uint32_t> queueFamily,
                                                          vk::CommandPoolCreateFlags commandPoolCreateFlags) const;
    [[nodiscard]] vk::UniqueFence createFence() const {
        return device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
    void recreateSwapChain();
    static void framebufferResizeCallback(void* userPointer, int width, int height);
    [[nodiscard]] BufferWithMemory createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                vk::MemoryPropertyFlags properties) const;
    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    [[nodiscard]] BufferWithMemory createVertexBuffer() const;
    [[nodiscard]] BufferWithMemory createIndexBuffer() const;
    template<typename CopyCommand, typename FlushBuffer>
    void singleTimeCommand(CopyCommand copyCommand, FlushBuffer flushBuffer) const;
    [[nodiscard]] vk::UniqueDescriptorSetLayout createDescriptorSetLayout() const;
    void copyViaStagingBuffer(const void *src, size_t size, const BufferWithMemory &dst) const;
    template<typename CopyCommandFactory>
    void copyViaStagingBuffer(const void *src, size_t size, CopyCommandFactory copyCommandFactory) const;
    [[nodiscard]] vk::UniqueCommandBuffer createTransferCommandBuffer() const;
    [[nodiscard]] ImageWithMemory createTextureImage() const;
    [[nodiscard]] ImageWithMemory createImage(uint32_t width, uint32_t height, vk::Format format,
                                              vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                                              vk::MemoryPropertyFlags properties) const;
    void transitionImageLayout(vk::Image image, vk::Format format, SwitchLayout switchLayout) const;
    [[nodiscard]] vk::UniqueImageView createImageView(vk::Image image, vk::Format format,
                                                      vk::ImageAspectFlags aspectFlags) const;
    [[nodiscard]] vk::UniqueSampler createTextureSampler() const;
    [[nodiscard]] vk::Format findSupportedFormat(vk::ArrayProxy<const vk::Format> candidates, vk::ImageTiling tiling,
                                                 vk::FormatFeatureFlags features) const;
    [[nodiscard]] vk::Format findDepthFormat() const {
        return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                                   vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }
    [[nodiscard]] vk::UniqueShaderModule createShaderModule(const std::vector<char> &code) const;

    std::chrono::high_resolution_clock::time_point startTime;
    GLFWWindow window;
    vk::UniqueInstance instance;
    vk::DynamicLoader dl;
    vk::DispatchLoaderDynamic dldi;
    debug::DynamicUniqueDebugUtilsMessengerEXT debugMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    QueueFamilyIndices queueFamilyIndices;
    vk::UniqueDevice device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::Queue transferQueue;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueCommandPool commandPool;
    vk::UniqueCommandPool transferCommandPool;
    vk::UniqueCommandBuffer transferCommandBuffer;
    BufferWithMemory vertexBuffer;
    BufferWithMemory indexBuffer;
    ImageWithMemory textureImage;
    vk::UniqueImageView textureImageView;
    vk::UniqueSampler textureSampler;
    std::array<vk::UniqueSemaphore, maxFramesInFlight> imageAvailableSemaphore;
    std::array<vk::UniqueSemaphore, maxFramesInFlight> renderFinishedSemaphore;
    std::array<vk::UniqueFence, maxFramesInFlight> inFlightFences;
    vk::Format depthFormat;
    vk::UniqueShaderModule vertShaderModule;
    vk::UniqueShaderModule fragShaderModule;
    SizeDependentResources res;
    size_t currentFrame;

    friend class SizeDependentResources;
};

#endif //VULKAN_TUTOR_BASEGRAPHICS_H
