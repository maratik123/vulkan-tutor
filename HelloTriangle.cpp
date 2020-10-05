#include "HelloTriangle.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <cstdint>

#include "utils.h"

namespace {
    constexpr std::array<const char *, 1> validationLayers {"VK_LAYER_KHRONOS_validation"};
    constexpr std::array<const char *, 1> deviceExtensions {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    constexpr std::array<vk::PhysicalDeviceType, 3> suitableDeviceTypesInPriority {
            vk::PhysicalDeviceType::eDiscreteGpu,
            vk::PhysicalDeviceType::eIntegratedGpu,
            vk::PhysicalDeviceType::eVirtualGpu
    };

    void checkValidationLayerSupport() {
        const auto &availableLayers = vk::enumerateInstanceLayerProperties();
        for (const auto &validationLayer : validationLayers) {
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
        const auto &availableExtensions = vk::enumerateInstanceExtensionProperties();
        for (const auto &requiredExtension : requiredExtensions) {
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
        const auto &availableExtensions = device.enumerateDeviceExtensionProperties();

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

    constexpr bool filterLog(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                   const vk::DebugUtilsMessageTypeFlagsEXT &messageType) {
        if (messageSeverity != vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
            return true;
        }
        return static_cast<bool>(messageType & allowedMessageTypes);
    }

    void debugCallback(
            vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            const vk::DebugUtilsMessageTypeFlagsEXT &messageType,
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

        const auto &requiredExtensions = GLFWWindow::requiredExtensions();
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

        const auto &debugInfo = createDebugInfo();
        createInfo.pNext = &debugInfo;
        return vk::createInstanceUnique(createInfo);
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
        const auto &it = std::find_if(availableFormats.cbegin(), availableFormats.cend(),
                                      [](const auto &availableFormat) {
                                          return availableFormat.format == vk::Format::eB8G8R8A8Srgb
                                                 && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
                                      });
        return it == availableFormats.cend() ? availableFormats.front() : *it;
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
        const auto &it = std::find(availablePresentModes.cbegin(), availablePresentModes.cend(),
                                   vk::PresentModeKHR::eMailbox);
        return it == availablePresentModes.cend() ? vk::PresentModeKHR::eFifo : vk::PresentModeKHR::eMailbox;
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
          graphicsQueue(logicalDevice->getQueue(queueFamilyIndices.graphicsFamily.value(), 0)),
          presentQueue(logicalDevice->getQueue(queueFamilyIndices.presentFamily.value(), 0)),
          swapChain(createSwapChain()),
          swapChainImages(getSwapChainImages()),
          swapChainImageViews(createSwapChainImageViews()),
          renderPass(createRenderPass()),
          pipelineLayout(createPipelineLayout()),
          graphicsPipeline(createGraphicsPipeline()),
          swapChainFramebuffers(createFramebuffers()),
          commandPool(createCommandPool()),
          commandBuffers(createCommandBuffers()),
          imageAvailableSemaphore({logicalDevice->createSemaphoreUnique({}),
                                   logicalDevice->createSemaphoreUnique({})}),
          renderFinishedSemaphore({logicalDevice->createSemaphoreUnique({}),
                                   logicalDevice->createSemaphoreUnique({})}),
          inFlightFences({createFence(),
                          createFence()}),
          imagesInFlight(createImageFenceReferences()) {
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
    const auto &queueFamilies = requestedDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto &queueFamily = queueFamilies[i];
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            result.graphicsFamily = i;
        }

        if (requestedDevice.getSurfaceSupportKHR(i, *surface)) {
            result.presentFamily = i;
        }

        if (result.isComplete()) {
            break;
        }
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
    const auto &swapChainSupport = querySwapChainSupport(requestedDevice);
    return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
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
    const auto &devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (auto deviceType : suitableDeviceTypesInPriority) {
        const auto &it = std::find_if(
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
    float queuePriority = 1.0;

    std::set<uint32_t> uniqueQueueFamilies{
            queueFamilyIndices.graphicsFamily.value(),
            queueFamilyIndices.presentFamily.value()
    };

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
    queueCreateInfos.reserve(uniqueQueueFamilies.size());

    for (uint32_t queueFamily : uniqueQueueFamilies) {
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
    const auto &swapChainSupport = querySwapChainSupport();

    const auto &surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    const auto &presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    const auto &extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    const std::array<const uint32_t, 2> queryFamilyIndices{
            queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.presentFamily.value()
    };

    const bool sameFamily = queueFamilyIndices.graphicsFamily == queueFamilyIndices.presentFamily;

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
                    sameFamily ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent,
                    sameFamily ? 0 : queryFamilyIndices.size(),
                    sameFamily ? nullptr : queryFamilyIndices.data(),
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

    for (const auto &image : swapChainImages) {
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
    const auto &vertShaderCode = readFile("shaders/shader.vert.spv");
    const auto &fragShaderCode = readFile("shaders/shader.frag.spv");

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
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
            {},
            0,
            nullptr,
            0,
            nullptr
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

vk::UniqueCommandPool HelloTriangle::createCommandPool() const {
    return logicalDevice->createCommandPoolUnique(vk::CommandPoolCreateInfo(
            {},
            queueFamilyIndices.graphicsFamily.value()
    ));
}

std::vector<vk::UniqueCommandBuffer> HelloTriangle::createCommandBuffers() const {
    std::vector<vk::UniqueCommandBuffer> result =
            logicalDevice->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
                    *commandPool,
                    vk::CommandBufferLevel::ePrimary,
                    swapChainFramebuffers.size()
            ));

    for (uint32_t i = 0; i < result.size(); ++i) {
        const vk::UniqueCommandBuffer &commandBuffer = result[i];

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
        commandBuffer->draw(3, 1, 0, 0);
        commandBuffer->endRenderPass();
        commandBuffer->end();
    }

    return std::move(result);
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
        logicalDevice->waitForFences({*imagesInFlight[imageIndex].value().get()}, VK_TRUE, UINT64_MAX);
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
        const auto &extent = window.getFramebufferSize();
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

void HelloTriangle::framebufferResizeCallback(void* userPointer, int width, int height) {
    auto *_this = reinterpret_cast<HelloTriangle *>(userPointer);
    _this->framebufferResized = true;
}
