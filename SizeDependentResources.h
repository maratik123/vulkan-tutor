#ifndef VULKAN_TUTOR_SIZEDEPENDENTRESOURCES_H
#define VULKAN_TUTOR_SIZEDEPENDENTRESOURCES_H

#include <vector>

#include "Common.h"

using OptRefUniqueFence = std::optional<std::reference_wrapper<vk::UniqueFence>>;

struct SwapChain {
    vk::UniqueSwapchainKHR swapChain{};
    vk::Format imageFormat{};
    vk::Extent2D extent{};
};

enum class AfterDrawAction {
    Noop,
    RecreateSwapChain
};

class BaseGraphics;

class SizeDependentResources {
    BaseGraphics &base;

public:
    explicit SizeDependentResources(BaseGraphics &base);
    SizeDependentResources &operator =(SizeDependentResources &&other) noexcept;
    AfterDrawAction drawFrame();

    bool framebufferResized;

private:
    [[nodiscard]] std::vector<OptRefUniqueFence> createImageFenceReferences() const;
    [[nodiscard]] std::vector<vk::Image> getSwapChainImages() const;
    [[nodiscard]] std::vector<vk::UniqueImageView> createSwapChainImageViews() const;
    [[nodiscard]] vk::UniquePipeline createGraphicsPipeline() const;
    [[nodiscard]] vk::UniqueRenderPass createRenderPass() const;
    [[nodiscard]] std::vector<vk::UniqueFramebuffer> createFramebuffers() const;
    [[nodiscard]] std::vector<vk::UniqueCommandBuffer> createCommandBuffers() const;
    [[nodiscard]] std::vector<BufferWithMemory> createUniformBuffers() const;
    void updateUniformBuffer(uint32_t imageIndex);
    [[nodiscard]] vk::UniqueDescriptorPool createDescriptorPool() const;
    [[nodiscard]] std::vector<vk::UniqueDescriptorSet> createDescriptorSets() const;
    [[nodiscard]] SwapChain createSwapChain() const;
    [[nodiscard]] vk::UniquePipelineLayout createPipelineLayout() const;
    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const;
    [[nodiscard]] vk::Device device() const;
    [[nodiscard]] ImageWithMemory createDepthImage() const;
    [[nodiscard]] vk::UniqueImageView createDepthImageView() const;

    SwapChain swapChain;
    ImageWithMemory depthImage;
    vk::UniqueImageView depthImageView;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::UniqueImageView> swapChainImageViews;
    vk::UniqueRenderPass renderPass;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;
    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;
    std::vector<BufferWithMemory> uniformBuffers;
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueDescriptorSet> descriptorSets;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    std::vector<OptRefUniqueFence> imagesInFlight;
};

#endif //VULKAN_TUTOR_SIZEDEPENDENTRESOURCES_H
