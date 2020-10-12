#include "HelloTriangle.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <cstdint>
#include "absl/container/flat_hash_set.h"

#include "utils.h"

namespace {
    struct Vertex {
        glm::vec2 pos;
        glm::vec3 color;

        static constexpr vk::VertexInputBindingDescription bindingDescription() {
            return {
                    0,
                    sizeof(Vertex),
                    vk::VertexInputRate::eVertex
            };
        }

        static constexpr std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
            return {{
                            {
                                    0,
                                    0,
                                    vk::Format::eR32G32Sfloat,
                                    offsetof(Vertex, pos)
                            },
                            {
                                    1,
                                    0,
                                    vk::Format::eR32G32B32Sfloat,
                                    offsetof(Vertex, color)
                            }
                    }};
        }
    };

    constexpr std::array<const char *, 1> validationLayers {"VK_LAYER_KHRONOS_validation"};
    constexpr std::array<const char *, 1> deviceExtensions {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    constexpr std::array<Vertex, 4> vertices{{
                                                     {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                                     {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                                                     {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                                                     {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
                                             }};
    constexpr std::array<uint16_t, 6> indices{{
                                                      0, 1, 2, 2, 3, 0
                                              }};

    constexpr uint64_t verticesSize = sizeof(decltype(vertices)::value_type) * vertices.size();
    constexpr uint64_t indicesSize = sizeof(decltype(indices)::value_type) * indices.size();

    constexpr std::array<vk::PhysicalDeviceType, 3> suitableDeviceTypesInPriority {
            vk::PhysicalDeviceType::eDiscreteGpu,
            vk::PhysicalDeviceType::eIntegratedGpu,
            vk::PhysicalDeviceType::eVirtualGpu
    };

    void checkValidationLayerSupport() {
        const auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const auto validationLayer : validationLayers) {
            if (std::none_of(
                    availableLayers.cbegin(), availableLayers.cend(),
                    [&validationLayer](const auto &availableLayer) {
                        return std::strcmp(validationLayer, availableLayer.layerName) == 0;
                    })) {
                throw vk::LayerNotPresentError(validationLayer);
            }
        }
    }

    void checkExtensions(const std::vector<const char *> &requiredExtensions) {
        const auto availableExtensions = vk::enumerateInstanceExtensionProperties();
        for (const auto requiredExtension : requiredExtensions) {
            if (std::none_of(
                    availableExtensions.cbegin(), availableExtensions.cend(),
                    [&requiredExtension](const auto &availableExtension) {
                        return std::strcmp(requiredExtension, availableExtension.extensionName) == 0;
                    })) {
                throw vk::ExtensionNotPresentError(requiredExtension);
            }
        }
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice &device) {
        const auto availableExtensions = device.enumerateDeviceExtensionProperties();

        return std::all_of(
                deviceExtensions.cbegin(), deviceExtensions.cend(),
                [&availableExtensions](const auto &requiredExtension) {
                    return std::any_of(
                            availableExtensions.cbegin(), availableExtensions.cend(),
                            [&requiredExtension](const auto &availableExtension) {
                                return std::strcmp(requiredExtension, availableExtension.extensionName) == 0;
                            });
                });
    }

    constexpr vk::DebugUtilsMessageTypeFlagsEXT allowedMessageTypes =
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;

    constexpr bool filterLog(const vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                             const vk::DebugUtilsMessageTypeFlagsEXT messageType) {
        if (messageSeverity != vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
            return true;
        }
        return static_cast<bool>(messageType & allowedMessageTypes);
    }

    void debugCallback(
            const vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            const vk::DebugUtilsMessageTypeFlagsEXT messageType,
            const vk::DebugUtilsMessengerCallbackDataEXT &callbackData) {
        if (filterLog(messageSeverity, messageType)) {
            std::cerr << print_time << vk::to_string(messageSeverity) << ' '
                      << vk::to_string(messageType)
                      << ' ' << callbackData.pMessage << std::endl;
        }
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) {
        debugCallback(
                static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity),
                static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageType),
                *pCallbackData
        );
        return VK_FALSE;
    }

    constexpr vk::DebugUtilsMessengerCreateInfoEXT createDebugInfo() {
        return vk::DebugUtilsMessengerCreateInfoEXT(
                {},
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                debugCallback,
                nullptr
        );
    }

    vk::UniqueInstance createInstance() {
        if (enableValidationLayers) {
            checkValidationLayerSupport();
        }

        vk::ApplicationInfo appInfo(
                "Hello Triangle",
                VK_MAKE_VERSION(1, 0, 0),
                "No engine",
                VK_MAKE_VERSION(1, 0, 0),
                VK_API_VERSION_1_0
        );

        const auto requiredExtensions = GLFWWindow::requiredExtensions();
        checkExtensions(requiredExtensions);

        vk::InstanceCreateInfo createInfo(
                {},
                &appInfo,
                enableValidationLayers ? validationLayers.size() : 0,
                enableValidationLayers ? validationLayers.data() : nullptr,
                requiredExtensions.size(),
                requiredExtensions.data()
        );

        if (!enableDebugMessenger) {
            return vk::createInstanceUnique(createInfo);
        }

        const auto debugInfo = createDebugInfo();
        createInfo.pNext = &debugInfo;
        return vk::createInstanceUnique(createInfo);
    }

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

HelloTriangle::HelloTriangle()
        : window(framebufferResizeCallback, this),
          instance(createInstance()),
          dl{},
          dldi(*instance, getVkGetInstanceProcAddr()),
          debugMessenger(setupDebugMessenger()),
          surface(window.createSurfaceUnique(instance.get())),
          physicalDevice(pickPhysicalDevice()),
          queueFamilyIndices(findQueueFamilies()),
          logicalDevice(createLogicalDevice()),
          graphicsQueue(logicalDevice->getQueue(*queueFamilyIndices.graphicsFamily, 0)),
          presentQueue(logicalDevice->getQueue(*queueFamilyIndices.presentFamily, 0)),
          transferQueue(logicalDevice->getQueue(*queueFamilyIndices.transferFamily, 0)),
          swapChain(createSwapChain()),
          swapChainImages(getSwapChainImages()),
          swapChainImageViews(createSwapChainImageViews()),
          renderPass(createRenderPass()),
          pipelineLayout(createPipelineLayout()),
          graphicsPipeline(createGraphicsPipeline()),
          swapChainFramebuffers(createFramebuffers()),
          commandPool(createCommandPool(queueFamilyIndices.graphicsFamily)),
          transferCommandPool(createCommandPool(queueFamilyIndices.transferFamily)),
          vertexBuffer(createVertexBuffer()),
          indexBuffer(createIndexBuffer()),
          commandBuffers(createCommandBuffers()),
          imageAvailableSemaphore({logicalDevice->createSemaphoreUnique({}),
                                   logicalDevice->createSemaphoreUnique({})}),
          renderFinishedSemaphore({logicalDevice->createSemaphoreUnique({}),
                                   logicalDevice->createSemaphoreUnique({})}),
          inFlightFences({createFence(),
                          createFence()}),
          imagesInFlight(createImageFenceReferences()),
          currentFrame(0),
          framebufferResized(false) {
    std::cout << print_time << "Initialized" << std::endl;
}

SwapChainSupportDetails HelloTriangle::querySwapChainSupport(const vk::PhysicalDevice &requestedDevice) const {
    return {
            requestedDevice.getSurfaceCapabilitiesKHR(*surface),
            requestedDevice.getSurfaceFormatsKHR(*surface),
            requestedDevice.getSurfacePresentModesKHR(*surface)
    };
}

QueueFamilyIndices HelloTriangle::findQueueFamilies(const vk::PhysicalDevice &requestedDevice) const {
    QueueFamilyIndices result{};
    const auto queueFamilies = requestedDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto queueFlags = queueFamilies[i].queueFlags;
        if (!result.graphicsFamily && (queueFlags & vk::QueueFlagBits::eGraphics)) {
            result.graphicsFamily = i;
        } else if (!result.transferFamily && (queueFlags & vk::QueueFlagBits::eTransfer)) {
            result.transferFamily = i;
        }

        if (!result.presentFamily && requestedDevice.getSurfaceSupportKHR(i, *surface)) {
            result.presentFamily = i;
        }

        if (result.isComplete()) {
            break;
        }
    }
    if (!result.transferFamily
        && result.graphicsFamily
        && queueFamilies[*result.graphicsFamily].queueFlags & vk::QueueFlagBits::eTransfer) {
        result.transferFamily = result.graphicsFamily;
    }
    return result;
}

bool HelloTriangle::isDeviceSuitable(const vk::PhysicalDevice &requestedDevice,
                                     vk::PhysicalDeviceType desiredDeviceType) const {
    if (requestedDevice.getProperties().deviceType != desiredDeviceType) {
        return false;
    }
    if (!requestedDevice.getFeatures().geometryShader) {
        return false;
    }
    if (!findQueueFamilies(requestedDevice).isComplete()) {
        return false;
    }
    if (!checkDeviceExtensionSupport(requestedDevice)) {
        return false;
    }
    const auto swapChainSupport = querySwapChainSupport(requestedDevice);
    return !(swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty());
}

PFN_vkGetInstanceProcAddr HelloTriangle::getVkGetInstanceProcAddr() const {
    return dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
}

HelloTriangle::DynamicUniqueDebugUtilsMessengerEXT HelloTriangle::setupDebugMessenger() const {
    if (!enableDebugMessenger) {
        return {};
    }

    return instance->createDebugUtilsMessengerEXTUnique<vk::DispatchLoaderDynamic>(
            createDebugInfo(), nullptr, dldi
    );
}

void HelloTriangle::run() {
    window.mainLoop([this]{ drawFrame(); });
    logicalDevice->waitIdle();
}

vk::PhysicalDevice HelloTriangle::pickPhysicalDevice() const {
    const auto devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (auto deviceType : suitableDeviceTypesInPriority) {
        const auto it = std::find_if(
                devices.cbegin(), devices.cend(),
                [this, deviceType](const auto &device) { return isDeviceSuitable(device, deviceType); }
        );
        if (it != devices.cend()) {
            return *it;
        }
    }
    throw std::runtime_error("failed to find a suitable GPU!");
}

HelloTriangle::~HelloTriangle() {
    std::cout << print_time << "Finalization" << std::endl;
}

vk::UniqueDevice HelloTriangle::createLogicalDevice() const {
    float queuePriority = 1.0f;

    absl::flat_hash_set<uint32_t> uniqueQueueFamilies{
            *queueFamilyIndices.graphicsFamily,
            *queueFamilyIndices.presentFamily,
            *queueFamilyIndices.transferFamily
    };

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
    queueCreateInfos.reserve(uniqueQueueFamilies.size());

    for (const auto queueFamily : uniqueQueueFamilies) {
        queueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo(
                {},
                queueFamily,
                1,
                &queuePriority
        ));
    }

    vk::PhysicalDeviceFeatures deviceFeatures {};

    return physicalDevice.createDeviceUnique(vk::DeviceCreateInfo(
            {},
            queueCreateInfos.size(),
            queueCreateInfos.data(),
            enableValidationLayers ? validationLayers.size() : 0,
            enableValidationLayers ? validationLayers.data() : nullptr,
            deviceExtensions.size(),
            deviceExtensions.data(),
            &deviceFeatures
    ));
}

SwapChain HelloTriangle::createSwapChain() const {
    const auto swapChainSupport = querySwapChainSupport();

    const auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    const auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    const auto extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    const absl::flat_hash_set<uint32_t> queryFamilyIndicesUnique{
            *queueFamilyIndices.graphicsFamily,
            *queueFamilyIndices.presentFamily,
            *queueFamilyIndices.transferFamily
    };

    const std::vector<uint32_t> queryFamilyIndices(queryFamilyIndicesUnique.cbegin(), queryFamilyIndicesUnique.cend());
    const bool exclusiveFamilyIndex = queryFamilyIndices.size() == 1;

    return {
            logicalDevice->createSwapchainKHRUnique(vk::SwapchainCreateInfoKHR(
                    {},
                    *surface,
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

std::vector<vk::UniqueImageView> HelloTriangle::createSwapChainImageViews() const {
    std::vector<vk::UniqueImageView> result{};
    result.reserve(swapChainImages.size());

    for (const auto image : swapChainImages) {
        result.emplace_back(logicalDevice->createImageViewUnique(vk::ImageViewCreateInfo(
                {},
                image,
                vk::ImageViewType::e2D,
                swapChain.imageFormat,
                {
                        vk::ComponentSwizzle::eIdentity,
                        vk::ComponentSwizzle::eIdentity,
                        vk::ComponentSwizzle::eIdentity,
                        vk::ComponentSwizzle::eIdentity
                },
                {
                        vk::ImageAspectFlagBits::eColor,
                        0,
                        1,
                        0,
                        1
                }
        )));
    }

    return result;
}

vk::UniquePipeline HelloTriangle::createGraphicsPipeline() const {
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

    const auto bindingDescription = Vertex::bindingDescription();
    const auto attributeDescriptions = Vertex::getAttributeDescriptions();

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
    return logicalDevice->createGraphicsPipelineUnique({}, vk::GraphicsPipelineCreateInfo(
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

vk::UniqueShaderModule HelloTriangle::createShaderModule(const std::vector<char> &code) const {
    return logicalDevice->createShaderModuleUnique(vk::ShaderModuleCreateInfo(
            {},
            code.size(),
            reinterpret_cast<const uint32_t *>(code.data())
    ));
}

vk::UniquePipelineLayout HelloTriangle::createPipelineLayout() const {
    return logicalDevice->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(
            {},
            0,
            nullptr,
            0,
            nullptr
    ));
}

vk::UniqueRenderPass HelloTriangle::createRenderPass() const {
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
    return logicalDevice->createRenderPassUnique(vk::RenderPassCreateInfo(
            {},
            1,
            &colorAttachment,
            1,
            &subpass,
            1,
            &dependency
    ));
}

std::vector<vk::UniqueFramebuffer> HelloTriangle::createFramebuffers() const {
    std::vector<vk::UniqueFramebuffer> result{};
    result.reserve(swapChainImageViews.size());

    for (const auto &imageView : swapChainImageViews) {
        result.emplace_back(logicalDevice->createFramebufferUnique(vk::FramebufferCreateInfo(
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

vk::UniqueCommandPool HelloTriangle::createCommandPool(std::optional<uint32_t> queueFamily) const {
    return logicalDevice->createCommandPoolUnique(vk::CommandPoolCreateInfo(
            {},
            *queueFamily
    ));
}

std::vector<vk::UniqueCommandBuffer> HelloTriangle::createCommandBuffers() const {
    auto commandBuffers_ = logicalDevice->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
            *commandPool,
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
        commandBuffer->bindVertexBuffers(0, {*vertexBuffer.buffer}, {0});
        commandBuffer->bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint16);
        commandBuffer->drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        commandBuffer->endRenderPass();
        commandBuffer->end();
    }

    return commandBuffers_;
}

void HelloTriangle::drawFrame() {
    logicalDevice->waitForFences({*inFlightFences[currentFrame]}, VK_TRUE, UINT64_MAX);
    uint32_t imageIndex;
    try {
        imageIndex = logicalDevice->acquireNextImageKHR(
                *swapChain.swapChain, UINT64_MAX, *imageAvailableSemaphore[currentFrame], {});
    } catch (const vk::OutOfDateKHRError &e) {
        recreateSwapChain();
        return;
    }
    if (imagesInFlight[imageIndex]) {
        logicalDevice->waitForFences({*imagesInFlight[imageIndex]->get()}, VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];
    std::array<vk::PipelineStageFlags, 1> waitStages{
            vk::PipelineStageFlagBits::eColorAttachmentOutput
    };
    logicalDevice->resetFences({*inFlightFences[currentFrame]});
    graphicsQueue.submit({vk::SubmitInfo(
            1,
            &*imageAvailableSemaphore[currentFrame],
            waitStages.data(),
            1,
            &*commandBuffers[imageIndex],
            1,
            &*renderFinishedSemaphore[currentFrame]
    )}, *inFlightFences[currentFrame]);
    vk::Result result;
    try {
        result = presentQueue.presentKHR(vk::PresentInfoKHR(
                1,
                &*renderFinishedSemaphore[currentFrame],
                1,
                &*swapChain.swapChain,
                &imageIndex
        ));
    } catch (const vk::OutOfDateKHRError &e) {
        recreateSwapChain();
    }
    if (result == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    currentFrame = (currentFrame + 1) % maxFramesInFlight;
}

void HelloTriangle::cleanupSwapChain() {
    imagesInFlight.clear();
    commandBuffers.clear();
    swapChainFramebuffers.clear();
    graphicsPipeline = {};
    pipelineLayout = {};
    renderPass = {};
    swapChainImageViews.clear();
    swapChainImages.clear();
    swapChain = {};
}

void HelloTriangle::recreateSwapChain() {
    while(true) {
        const auto extent = window.getFramebufferSize();
        if (extent.width != 0 && extent.height != 0) {
            break;
        }
        GLFWWindow::waitEvents();
    }

    logicalDevice->waitIdle();

    cleanupSwapChain();

    swapChain = createSwapChain();
    swapChainImages = getSwapChainImages();
    swapChainImageViews = createSwapChainImageViews();
    renderPass = createRenderPass();
    pipelineLayout = createPipelineLayout();
    graphicsPipeline = createGraphicsPipeline();
    swapChainFramebuffers = createFramebuffers();
    commandBuffers = createCommandBuffers();
    imagesInFlight = createImageFenceReferences();
}

vk::Extent2D HelloTriangle::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    auto actualExtent = window.getFramebufferSize();
    return actualExtent
            .setWidth(std::clamp(actualExtent.width,
                                 capabilities.minImageExtent.width, capabilities.maxImageExtent.width))
            .setHeight(std::clamp(actualExtent.height,
                                  capabilities.minImageExtent.height, capabilities.maxImageExtent.height));
}

void HelloTriangle::framebufferResizeCallback(void* userPointer, int /*width*/, int /*height*/) {
    auto *_this = reinterpret_cast<HelloTriangle *>(userPointer);
    _this->framebufferResized = true;
}

vk::UniqueBuffer HelloTriangle::createDeviceBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) const {
    return logicalDevice->createBufferUnique(vk::BufferCreateInfo{
            {},
            size,
            usage,
            vk::SharingMode::eExclusive
    });
}

vk::UniqueDeviceMemory HelloTriangle::allocateVertexBufferMemory(vk::Buffer buffer,
                                                                 vk::MemoryPropertyFlags properties) const {
    const auto memRequirements = logicalDevice->getBufferMemoryRequirements(buffer);
    auto vertexBufferMemory = logicalDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
            memRequirements.size,
            findMemoryType(memRequirements.memoryTypeBits, properties)
    ));
    logicalDevice->bindBufferMemory(buffer, *vertexBufferMemory, 0);

    return vertexBufferMemory;
}

uint32_t HelloTriangle::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    const auto memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

BufferWithMemory HelloTriangle::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                             vk::MemoryPropertyFlags properties) const {
    auto buffer = createDeviceBuffer(size, usage);

    return {
            allocateVertexBufferMemory(*buffer, properties),
            std::move(buffer)
    };
}

BufferWithMemory HelloTriangle::createVertexBuffer() const {
    auto stagingBuffer = createBuffer(
            verticesSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached);

    void *data = logicalDevice->mapMemory(*stagingBuffer.bufferMemory, 0, verticesSize);
    std::memcpy(data, vertices.data(), static_cast<size_t>(verticesSize));

    auto result = createBuffer(
            verticesSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(stagingBuffer, result, verticesSize);
    logicalDevice->unmapMemory(*stagingBuffer.bufferMemory);
    return result;
}

BufferWithMemory HelloTriangle::createIndexBuffer() const {
    auto stagingBuffer = createBuffer(
            indicesSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached);

    void *data = logicalDevice->mapMemory(*stagingBuffer.bufferMemory, 0, indicesSize);
    std::memcpy(data, indices.data(), static_cast<size_t>(indicesSize));

    auto result = createBuffer(
            indicesSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(stagingBuffer, result, indicesSize);
    logicalDevice->unmapMemory(*stagingBuffer.bufferMemory);
    return result;
}

void HelloTriangle::copyBuffer(const BufferWithMemory &srcBuffer, const BufferWithMemory &dstBuffer,
                               vk::DeviceSize size) const {
    auto commandBuffer = std::move(logicalDevice->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo(
                    *transferCommandPool,
                    vk::CommandBufferLevel::ePrimary,
                    1
            )
    ).front());

    commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    commandBuffer->copyBuffer(*srcBuffer.buffer, *dstBuffer.buffer, {
            vk::BufferCopy(0, 0, size)
    });
    commandBuffer->end();
    logicalDevice->flushMappedMemoryRanges({vk::MappedMemoryRange(*srcBuffer.bufferMemory, 0, VK_WHOLE_SIZE)});
    transferQueue.submit({
                                 vk::SubmitInfo({}, {}, {}, 1, &*commandBuffer)
                         }, {});
    transferQueue.waitIdle();
}
