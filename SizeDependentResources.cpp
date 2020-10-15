#include "SizeDependentResources.h"

#include <unordered_set>

#include "BaseGraphics.h"
#include "ShaderObjects.h"

namespace {
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
        const auto it = std::find_if(availableFormats.cbegin(), availableFormats.cend(),
                                     [](const auto &availableFormat) {
                                         return availableFormat.format == vk::Format::eB8G8R8A8Srgb
                                                && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
                                     });
        return it != availableFormats.cend() ? *it : availableFormats.front();
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
        const auto it = std::find(availablePresentModes.cbegin(), availablePresentModes.cend(),
                                  vk::PresentModeKHR::eMailbox);
        return it != availablePresentModes.cend() ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }
}

SizeDependentResources::SizeDependentResources(BaseGraphics &base)
        : base(base),
          logicalDevice(*base.logicalDevice),
          framebufferResized(false),
          swapChain(createSwapChain()),
          swapChainImages(getSwapChainImages()),
          swapChainImageViews(createSwapChainImageViews()),
          renderPass(createRenderPass()),
          pipelineLayout(createPipelineLayout()),
          graphicsPipeline(createGraphicsPipeline()),
          swapChainFramebuffers(createFramebuffers()),
          uniformBuffers(createUniformBuffers()),
          descriptorPool(createDescriptorPool()),
          descriptorSets(createDescriptorSets()),
          commandBuffers(createCommandBuffers()),
          imagesInFlight(createImageFenceReferences()) {
}

[[nodiscard]] std::vector<OptRefUniqueFence> SizeDependentResources::createImageFenceReferences() const {
    return std::vector<OptRefUniqueFence>(swapChainImages.size());
}

[[nodiscard]] std::vector<vk::Image> SizeDependentResources::getSwapChainImages() const {
    return logicalDevice.getSwapchainImagesKHR(*swapChain.swapChain);
}

std::vector<vk::UniqueImageView> SizeDependentResources::createSwapChainImageViews() const {
    std::vector<vk::UniqueImageView> result;
    result.reserve(swapChainImages.size());

    for (const auto image : swapChainImages) {
        result.emplace_back(base.createImageView(image, swapChain.imageFormat));
    }

    return result;
}

vk::UniquePipeline SizeDependentResources::createGraphicsPipeline() const {
    const auto vertShaderCode = readFile("shaders/shader.vert.spv");
    const auto fragShaderCode = readFile("shaders/shader.frag.spv");

    vk::UniqueShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    vk::UniqueShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageCreateInfo(
            {},
            vk::ShaderStageFlagBits::eVertex,
            *vertShaderModule,
            "main"
    );
    vk::PipelineShaderStageCreateInfo fragShaderStageCreateInfo(
            {},
            vk::ShaderStageFlagBits::eFragment,
            *fragShaderModule,
            "main"
    );
    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages {
            vertShaderStageCreateInfo, fragShaderStageCreateInfo
    };

    const auto bindingDescription = so::Vertex::bindingDescription();
    const auto attributeDescriptions = so::Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
            {},
            1,
            &bindingDescription,
            attributeDescriptions.size(),
            attributeDescriptions.data()
    );
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
            {},
            vk::PrimitiveTopology::eTriangleList,
            VK_FALSE
    );
    vk::Viewport viewport(
            0.0f,
            0.0f,
            static_cast<float>(swapChain.extent.width),
            static_cast<float>(swapChain.extent.height),
            0.0f,
            1.0f
    );
    vk::Rect2D scissor(
            {0, 0},
            swapChain.extent
    );

    vk::PipelineViewportStateCreateInfo viewportState(
            {},
            1,
            &viewport,
            1,
            &scissor
    );
    vk::PipelineRasterizationStateCreateInfo rasterizer(
            {},
            VK_FALSE,
            VK_FALSE,
            vk::PolygonMode::eFill,
            vk::CullModeFlagBits::eBack,
            vk::FrontFace::eClockwise,
            VK_FALSE,
            0.0f,
            0.0f,
            0.0f,
            1.0f
    );
    vk::PipelineMultisampleStateCreateInfo multisampling(
            {},
            vk::SampleCountFlagBits::e1,
            VK_FALSE,
            1.0f,
            nullptr,
            VK_FALSE,
            VK_FALSE
    );
    vk::PipelineColorBlendAttachmentState colorBlendAttachment(
            VK_FALSE,
            vk::BlendFactor::eOne,
            vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::BlendFactor::eOne,
            vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR
            | vk::ColorComponentFlagBits::eG
            | vk::ColorComponentFlagBits::eB
            | vk::ColorComponentFlagBits::eA
    );
    vk::PipelineColorBlendStateCreateInfo colorBlending(
            {},
            VK_FALSE,
            vk::LogicOp::eCopy,
            1,
            &colorBlendAttachment,
            {0.0f, 0.0f, 0.0f, 0.0f}
    );
    return logicalDevice.createGraphicsPipelineUnique({}, vk::GraphicsPipelineCreateInfo(
            {},
            2,
            shaderStages.data(),
            &vertexInputInfo,
            &inputAssembly,
            nullptr,
            &viewportState,
            &rasterizer,
            &multisampling,
            nullptr,
            &colorBlending,
            nullptr,
            *pipelineLayout,
            *renderPass,
            0,
            {},
            -1
    ));
}

vk::UniqueRenderPass SizeDependentResources::createRenderPass() const {
    vk::AttachmentDescription colorAttachment(
            {},
            swapChain.imageFormat,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::ePresentSrcKHR
    );
    vk::AttachmentReference colorAttachmentReference(
            0,
            vk::ImageLayout::eColorAttachmentOptimal
    );
    vk::SubpassDescription subpass(
            {},
            vk::PipelineBindPoint::eGraphics,
            0,
            nullptr,
            1,
            &colorAttachmentReference
    );
    vk::SubpassDependency dependency(
            VK_SUBPASS_EXTERNAL,
            0,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {},
            vk::AccessFlagBits::eColorAttachmentWrite
    );
    return logicalDevice.createRenderPassUnique(vk::RenderPassCreateInfo(
            {},
            1,
            &colorAttachment,
            1,
            &subpass,
            1,
            &dependency
    ));
}


std::vector<vk::UniqueFramebuffer> SizeDependentResources::createFramebuffers() const {
    std::vector<vk::UniqueFramebuffer> result;
    result.reserve(swapChainImageViews.size());

    for (const auto &imageView : swapChainImageViews) {
        result.emplace_back(logicalDevice.createFramebufferUnique(vk::FramebufferCreateInfo(
                {},
                *renderPass,
                1,
                &*imageView,
                swapChain.extent.width,
                swapChain.extent.height,
                1
        )));
    }

    return result;
}

std::vector<vk::UniqueCommandBuffer> SizeDependentResources::createCommandBuffers() const {
    auto commandBuffers_ = logicalDevice.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
            *base.commandPool,
            vk::CommandBufferLevel::ePrimary,
            swapChainFramebuffers.size()
    ));

    for (uint32_t i = 0; i < commandBuffers_.size(); ++i) {
        const auto &commandBuffer = commandBuffers_[i];

        commandBuffer->begin(vk::CommandBufferBeginInfo(
                {},
                nullptr
        ));

        vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4>{
                0.0f, 0.0f, 0.0f, 1.0f
        }));

        commandBuffer->beginRenderPass(vk::RenderPassBeginInfo(
                *renderPass,
                *swapChainFramebuffers[i],
                vk::Rect2D(
                        {0, 0},
                        swapChain.extent
                ),
                1,
                &clearColor
        ), vk::SubpassContents::eInline);
        commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffer->bindVertexBuffers(0, {*base.vertexBuffer.buffer}, {0});
        commandBuffer->bindIndexBuffer(*base.indexBuffer.buffer, 0, vk::IndexType::eUint16);
        commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, {*descriptorSets[i]},
                                          {});
        commandBuffer->drawIndexed(static_cast<uint32_t>(so::indices.size()), 1, 0, 0, 0);
        commandBuffer->endRenderPass();
        commandBuffer->end();
    }

    return commandBuffers_;
}

AfterDrawAction SizeDependentResources::drawFrame() {
    logicalDevice.waitForFences({*base.inFlightFences[base.currentFrame]}, VK_TRUE, UINT64_MAX);
    uint32_t imageIndex;
    try {
        imageIndex = logicalDevice.acquireNextImageKHR(
                *swapChain.swapChain, UINT64_MAX, *base.imageAvailableSemaphore[base.currentFrame], {});
    } catch (const vk::OutOfDateKHRError &e) {
        return AfterDrawAction::RecreateSwapChain;
    }
    if (imagesInFlight[imageIndex]) {
        logicalDevice.waitForFences({*imagesInFlight[imageIndex]->get()}, VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = base.inFlightFences[base.currentFrame];
    std::array<vk::PipelineStageFlags, 1> waitStages{
            vk::PipelineStageFlagBits::eColorAttachmentOutput
    };
    logicalDevice.resetFences({*base.inFlightFences[base.currentFrame]});
    updateUniformBuffer(imageIndex);
    base.graphicsQueue.submit({vk::SubmitInfo(
            1,
            &*base.imageAvailableSemaphore[base.currentFrame],
            waitStages.data(),
            1,
            &*commandBuffers[imageIndex],
            1,
            &*base.renderFinishedSemaphore[base.currentFrame]
    )}, *base.inFlightFences[base.currentFrame]);
    vk::Result result;
    try {
        result = base.presentQueue.presentKHR(vk::PresentInfoKHR(
                1,
                &*base.renderFinishedSemaphore[base.currentFrame],
                1,
                &*swapChain.swapChain,
                &imageIndex
        ));
    } catch (const vk::OutOfDateKHRError &e) {
        return AfterDrawAction::RecreateSwapChain;
    }
    if (result == vk::Result::eSuboptimalKHR || base.res.framebufferResized) {
        return AfterDrawAction::RecreateSwapChain;
    }
    base.currentFrame = (base.currentFrame + 1) % BaseGraphics::maxFramesInFlight;
    return AfterDrawAction::Noop;
}

SizeDependentResources &SizeDependentResources::operator =(SizeDependentResources &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    assert(&base == &other.base);
    imagesInFlight.clear();
    commandBuffers.clear();
    descriptorSets.clear();
    descriptorPool.reset();
    uniformBuffers.clear();
    swapChainFramebuffers.clear();
    graphicsPipeline.reset();
    pipelineLayout.reset();
    renderPass.reset();
    swapChainImageViews.clear();
    swapChainImages.clear();

    framebufferResized = other.framebufferResized;
    swapChain = std::move(other.swapChain);

    swapChainImages = std::move(other.swapChainImages);
    swapChainImageViews = std::move(other.swapChainImageViews);
    renderPass = std::move(other.renderPass);
    pipelineLayout = std::move(other.pipelineLayout);
    graphicsPipeline = std::move(other.graphicsPipeline);
    swapChainFramebuffers = std::move(other.swapChainFramebuffers);
    uniformBuffers = std::move(other.uniformBuffers);
    descriptorPool = std::move(other.descriptorPool);
    descriptorSets = std::move(other.descriptorSets);
    commandBuffers = std::move(other.commandBuffers);
    imagesInFlight = std::move(other.imagesInFlight);

    return *this;
}


std::vector<BufferWithMemory> SizeDependentResources::createUniformBuffers() const {
    const vk::DeviceSize bufferSize = sizeof(so::UnifiedBufferObject);

    std::vector<BufferWithMemory> uniformBuffers_;
    uniformBuffers_.reserve(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); ++i) {
        uniformBuffers_.emplace_back(base.createBuffer(
                bufferSize, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal
        ));
    }
    return uniformBuffers_;
}

void SizeDependentResources::updateUniformBuffer(uint32_t imageIndex) {
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float> duration = currentTime - base.startTime;
    auto proj = glm::perspective(
            glm::radians(45.0f),
            swapChain.extent.width / static_cast<float>(swapChain.extent.height),
            0.1f,
            10.0f);
    proj[1][1] *= -1;
    so::UnifiedBufferObject ubo {
            glm::rotate(
                    glm::mat4(1.0f),
                    duration.count() * glm::radians(90.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f)),
            glm::lookAt(
                    glm::vec3(2.0f, 2.0f, 2.0f),
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f)),
            proj
    };
    base.copyViaStagingBuffer(&ubo, sizeof(ubo), uniformBuffers[imageIndex]);
}

vk::UniqueDescriptorPool SizeDependentResources::createDescriptorPool() const {
    std::array<vk::DescriptorPoolSize, 2> poolSizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapChainImages.size()),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, swapChainImages.size())
    };
    return logicalDevice.createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            static_cast<uint32_t>(swapChainImages.size()),
            poolSizes.size(),
            poolSizes.data()
    ));
}

std::vector<vk::UniqueDescriptorSet> SizeDependentResources::createDescriptorSets() const {
    std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), *base.descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo(
            *descriptorPool,
            static_cast<uint32_t>(swapChainImages.size()),
            layouts.data()
    );

    std::vector<vk::UniqueDescriptorSet> descriptorSets_ = logicalDevice.allocateDescriptorSetsUnique(allocInfo);

    for (size_t i = 0; i < swapChainImages.size(); ++i) {
        vk::DescriptorBufferInfo bufferInfo(
                *uniformBuffers[i].buffer,
                0,
                sizeof(so::UnifiedBufferObject)
        );

        vk::DescriptorImageInfo imageInfo(
                *base.textureSampler,
                *base.textureImageView,
                vk::ImageLayout::eShaderReadOnlyOptimal
        );

        logicalDevice.updateDescriptorSets(
                {
                        vk::WriteDescriptorSet(
                                *descriptorSets_[i],
                                0,
                                0,
                                1,
                                vk::DescriptorType::eUniformBuffer,
                                nullptr,
                                &bufferInfo,
                                nullptr
                        ),
                        vk::WriteDescriptorSet(
                                *descriptorSets_[i],
                                1,
                                0,
                                1,
                                vk::DescriptorType::eCombinedImageSampler,
                                &imageInfo,
                                nullptr,
                                nullptr
                        )
                }, {});
    }

    return descriptorSets_;
}


SwapChain SizeDependentResources::createSwapChain() const {
    const auto swapChainSupport = base.querySwapChainSupport();

    const auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    const auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    const auto extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    const std::unordered_set<uint32_t> queryFamilyIndicesUnique{
            *base.queueFamilyIndices.graphicsFamily,
            *base.queueFamilyIndices.presentFamily,
            *base.queueFamilyIndices.transferFamily
    };

    const std::vector<uint32_t> queryFamilyIndices(queryFamilyIndicesUnique.cbegin(), queryFamilyIndicesUnique.cend());
    const bool exclusiveFamilyIndex = queryFamilyIndices.size() == 1;

    return {
            logicalDevice.createSwapchainKHRUnique(vk::SwapchainCreateInfoKHR(
                    {},
                    *base.surface,
                    imageCount,
                    surfaceFormat.format,
                    surfaceFormat.colorSpace,
                    extent,
                    1,
                    vk::ImageUsageFlagBits::eColorAttachment,
                    exclusiveFamilyIndex ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent,
                    exclusiveFamilyIndex ? 0 : queryFamilyIndices.size(),
                    exclusiveFamilyIndex ? nullptr : queryFamilyIndices.data(),
                    swapChainSupport.capabilities.currentTransform,
                    vk::CompositeAlphaFlagBitsKHR::eOpaque,
                    presentMode,
                    VK_TRUE
            )),
            surfaceFormat.format,
            extent
    };
}

vk::UniqueShaderModule SizeDependentResources::createShaderModule(const std::vector<char> &code) const {
    return logicalDevice.createShaderModuleUnique(vk::ShaderModuleCreateInfo(
            {},
            code.size(),
            reinterpret_cast<const uint32_t *>(code.data())
    ));
}

vk::UniquePipelineLayout SizeDependentResources::createPipelineLayout() const {
    return logicalDevice.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(
            {},
            1,
            &*base.descriptorSetLayout,
            0,
            nullptr
    ));
}

vk::Extent2D SizeDependentResources::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    auto actualExtent = base.window.getFramebufferSize();
    return actualExtent
            .setWidth(std::clamp(actualExtent.width,
                                 capabilities.minImageExtent.width, capabilities.maxImageExtent.width))
            .setHeight(std::clamp(actualExtent.height,
                                  capabilities.minImageExtent.height, capabilities.maxImageExtent.height));
}
