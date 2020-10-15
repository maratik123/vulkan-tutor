#ifndef VULKAN_TUTOR_SIZEDEPENDENTRESOURCES_H
#define VULKAN_TUTOR_SIZEDEPENDENTRESOURCES_H

#include <vector>

#include "Common.h"

class BaseGraphics;

enum class AfterDrawAction {
    Noop,
    RecreateSwapChain
};

class SizeDependentResources {
    BaseGraphics &base;
    const vk::Device logicalDevice;

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
    [[nodiscard]] vk::UniqueShaderModule createShaderModule(const std::vector<char> &code) const;
    [[nodiscard]] vk::UniquePipelineLayout createPipelineLayout() const;
    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const;

    SwapChain swapChain;
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
