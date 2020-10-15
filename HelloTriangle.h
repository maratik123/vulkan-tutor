#ifndef VULKAN_TUTOR_HELLOTRIANGLE_H
#define VULKAN_TUTOR_HELLOTRIANGLE_H

#include <optional>
#include <chrono>

#include "GLFWInclude.h"

#include "GLFWWindow.h"
#include "debug.h"

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

class HelloTriangle {
public:
    HelloTriangle();
    ~HelloTriangle();
    void run();

private:
    using OptRefUniqueFence = std::optional<std::reference_wrapper<vk::UniqueFence>>;
    static constexpr int maxFramesInFlight = 2;

    [[nodiscard]] PFN_vkGetInstanceProcAddr getVkGetInstanceProcAddr() const;
    [[nodiscard]] vk::PhysicalDevice pickPhysicalDevice() const;
    [[nodiscard]] vk::UniqueDevice createLogicalDevice() const;
    [[nodiscard]] bool isDeviceSuitable(
            const vk::PhysicalDevice &requestedDevice,
            vk::PhysicalDeviceType desiredDeviceType) const;
    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice &requestedDevice) const;
    [[nodiscard]] QueueFamilyIndices findQueueFamilies() const { return findQueueFamilies(physicalDevice); }
    [[nodiscard]] SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice &requestedDevice) const;
    [[nodiscard]] SwapChainSupportDetails querySwapChainSupport() const {
        return querySwapChainSupport(physicalDevice);
    }
    [[nodiscard]] SwapChain createSwapChain() const;
    [[nodiscard]] std::vector<vk::UniqueImageView> createSwapChainImageViews() const;
    [[nodiscard]] vk::UniquePipeline createGraphicsPipeline() const;
    [[nodiscard]] vk::UniqueShaderModule createShaderModule(const std::vector<char> &code) const;
    [[nodiscard]] vk::UniquePipelineLayout createPipelineLayout() const;
    [[nodiscard]] vk::UniqueRenderPass createRenderPass() const;
    [[nodiscard]] std::vector<vk::UniqueFramebuffer> createFramebuffers() const;
    [[nodiscard]] vk::UniqueCommandPool createCommandPool(std::optional<uint32_t> queueFamily,
                                                          vk::CommandPoolCreateFlags commandPoolCreateFlags) const;
    [[nodiscard]] std::vector<vk::UniqueCommandBuffer> createCommandBuffers() const;
    [[nodiscard]] vk::UniqueFence createFence() const {
        return logicalDevice->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
    [[nodiscard]] std::vector<OptRefUniqueFence> createImageFenceReferences() const {
        return std::vector<HelloTriangle::OptRefUniqueFence>(swapChainImages.size());
    }
    [[nodiscard]] std::vector<vk::Image> getSwapChainImages() const {
        return logicalDevice->getSwapchainImagesKHR(*swapChain.swapChain);
    }
    void cleanupSwapChain();
    void recreateSwapChain();
    void drawFrame();
    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const;
    static void framebufferResizeCallback(void* userPointer, int width, int height);
    [[nodiscard]] BufferWithMemory createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                vk::MemoryPropertyFlags properties) const;
    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    [[nodiscard]] BufferWithMemory createVertexBuffer() const;
    [[nodiscard]] BufferWithMemory createIndexBuffer() const;
    template<typename CopyCommand, typename FlushBuffer>
    void singleTimeCommand(CopyCommand copyCommand, FlushBuffer flushBuffer) const;
    [[nodiscard]] vk::UniqueDescriptorSetLayout createDescriptorSetLayout() const;
    [[nodiscard]] std::vector<BufferWithMemory> createUniformBuffers() const;
    void updateUniformBuffer(uint32_t imageIndex);
    [[nodiscard]] vk::UniqueDescriptorPool createDescriptorPool() const;
    [[nodiscard]] std::vector<vk::UniqueDescriptorSet> createDescriptorSets() const;
    void copyViaStagingBuffer(const void *src, size_t size, const BufferWithMemory &dst) const;
    template<typename CopyCommandFactory>
    void copyViaStagingBuffer(const void *src, size_t size, CopyCommandFactory copyCommandFactory) const;
    [[nodiscard]] vk::UniqueCommandBuffer createTransferCommandBuffer() const;
    [[nodiscard]] ImageWithMemory createTextureImage() const;
    [[nodiscard]] ImageWithMemory createImage(uint32_t width, uint32_t height, vk::Format format,
                                              vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                                              vk::MemoryPropertyFlags properties) const;
    void transitionImageLayout(vk::Image image, SwitchLayout switchLayout) const;
    [[nodiscard]] vk::UniqueImageView createImageView(vk::Image image, vk::Format format) const;
    [[nodiscard]] vk::UniqueSampler createTextureSampler() const;

    std::chrono::high_resolution_clock::time_point startTime;
    GLFWWindow window;
    vk::UniqueInstance instance;
    vk::DynamicLoader dl;
    vk::DispatchLoaderDynamic dldi;
    debug::DynamicUniqueDebugUtilsMessengerEXT debugMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    QueueFamilyIndices queueFamilyIndices;
    vk::UniqueDevice logicalDevice;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::Queue transferQueue;
    SwapChain swapChain;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::UniqueImageView> swapChainImageViews;
    vk::UniqueRenderPass renderPass;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;
    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;
    vk::UniqueCommandPool commandPool;
    vk::UniqueCommandPool transferCommandPool;
    vk::UniqueCommandBuffer transferCommandBuffer;
    BufferWithMemory vertexBuffer;
    BufferWithMemory indexBuffer;
    std::vector<BufferWithMemory> uniformBuffers;
    ImageWithMemory textureImage;
    vk::UniqueImageView textureImageView;
    vk::UniqueSampler textureSampler;
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueDescriptorSet> descriptorSets;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    std::array<vk::UniqueSemaphore, maxFramesInFlight> imageAvailableSemaphore;
    std::array<vk::UniqueSemaphore, maxFramesInFlight> renderFinishedSemaphore;
    std::array<vk::UniqueFence, maxFramesInFlight> inFlightFences;
    std::vector<OptRefUniqueFence> imagesInFlight;
    size_t currentFrame;
    bool framebufferResized;
};

#endif //VULKAN_TUTOR_HELLOTRIANGLE_H
