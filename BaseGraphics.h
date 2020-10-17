#ifndef VULKAN_TUTOR_BASEGRAPHICS_H
#define VULKAN_TUTOR_BASEGRAPHICS_H

#include <optional>
#include <chrono>

#include "GLFWInclude.h"

#include "GLFWWindow.h"
#include "Debug.h"
#include "SizeDependentResources.h"
#include "Model.h"

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

struct ModelBuffers {
    BufferWithMemory vertexBuffer{};
    BufferWithMemory indexBuffer{};
    size_t indicesCount{};
};

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
    [[nodiscard]] BufferWithMemory createVertexBuffer(const Model &model) const;
    [[nodiscard]] BufferWithMemory createIndexBuffer(const Model &model) const;
    template<typename CopyCommand, typename FlushBuffer>
    void singleTimeCommand(vk::CommandPool commandPool, CopyCommand copyCommand, FlushBuffer flushBuffer) const;
    [[nodiscard]] vk::UniqueDescriptorSetLayout createDescriptorSetLayout() const;
    void copyViaStagingBuffer(const void *src, size_t size, const BufferWithMemory &dst) const;
    template<typename CopyCommandFactory>
    void copyViaStagingBuffer(const void *src, size_t size, CopyCommandFactory copyCommandFactory) const;
    [[nodiscard]] vk::UniqueCommandBuffer createCommandBuffer(vk::CommandPool commandPool) const;
    [[nodiscard]] TextureImage createTextureImage() const;
    [[nodiscard]] ImageWithMemory createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                                              vk::SampleCountFlagBits numSamples, vk::Format format,
                                              vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                                              vk::MemoryPropertyFlags properties) const;
    void transitionImageLayout(vk::Image image, vk::Format format, SwitchLayout switchLayout, uint32_t mipLevels) const;
    void generateMipmaps(vk::Image image, vk::Format imageFormat, uint32_t texWidth, uint32_t texHeight,
                         uint32_t mipLevels) const;
    [[nodiscard]] vk::UniqueImageView createImageView(vk::Image image, vk::Format format,
                                                      vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const;
    [[nodiscard]] vk::UniqueSampler createTextureSampler() const;
    [[nodiscard]] vk::Format findSupportedFormat(vk::ArrayProxy<const vk::Format> candidates, vk::ImageTiling tiling,
                                                 vk::FormatFeatureFlags features) const;
    [[nodiscard]] vk::Format findDepthFormat() const {
        return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                                   vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }
    [[nodiscard]] vk::UniqueShaderModule createShaderModule(const std::vector<char> &code) const;
    [[nodiscard]] ModelBuffers createModelBuffers() const;
    [[nodiscard]] vk::SampleCountFlagBits getMaxUsableSampleCount() const;

    std::chrono::high_resolution_clock::time_point startTime;
    GLFWWindow window;
    vk::UniqueInstance instance;
    vk::DynamicLoader dl;
    vk::DispatchLoaderDynamic dldi;
    debug::DynamicUniqueDebugUtilsMessengerEXT debugMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::SampleCountFlagBits msaaSamples;
    QueueFamilyIndices queueFamilyIndices;
    vk::UniqueDevice device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::Queue transferQueue;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueCommandPool graphicsCommandPool;
    vk::UniqueCommandPool transferCommandPool;
    ModelBuffers modelBuffers;
    TextureImage textureImage;
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
