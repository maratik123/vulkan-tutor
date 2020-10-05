#ifndef VULKAN_TUTOR_HELLOTRIANGLE_H
#define VULKAN_TUTOR_HELLOTRIANGLE_H

#include <optional>
#include <vector>

#include "GLFWWindow.h"

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily{};
    std::optional<uint32_t> presentFamily{};

    [[nodiscard]] constexpr bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
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

class HelloTriangle {
public:
    HelloTriangle();
    ~HelloTriangle();
    void run();

private:
    using DynamicUniqueDebugUtilsMessengerEXT = vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>;
    using OptRefUniqueFence = std::optional<std::reference_wrapper<vk::UniqueFence>>;
    static constexpr int maxFramesInFlight = 2;

    [[nodiscard]] DynamicUniqueDebugUtilsMessengerEXT setupDebugMessenger() const;
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
    [[nodiscard]] vk::UniqueCommandPool createCommandPool() const;
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

    GLFWWindow window;
    vk::UniqueInstance instance;
    vk::DynamicLoader dl;
    vk::DispatchLoaderDynamic dldi;
    DynamicUniqueDebugUtilsMessengerEXT debugMessenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    QueueFamilyIndices queueFamilyIndices;
    vk::UniqueDevice logicalDevice;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    SwapChain swapChain;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::UniqueImageView> swapChainImageViews;
    vk::UniqueRenderPass renderPass;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;
    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;
    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    std::array<vk::UniqueSemaphore, maxFramesInFlight> imageAvailableSemaphore;
    std::array<vk::UniqueSemaphore, maxFramesInFlight> renderFinishedSemaphore;
    std::array<vk::UniqueFence, maxFramesInFlight> inFlightFences;
    std::vector<OptRefUniqueFence> imagesInFlight;
    size_t currentFrame = 0;
    bool framebufferResized = false;
};

#endif //VULKAN_TUTOR_HELLOTRIANGLE_H
