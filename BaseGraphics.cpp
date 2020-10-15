#include "BaseGraphics.h"

#include <unordered_set>
#include <iostream>

#include "ShaderObjects.h"
#include "Image.h"

namespace {
    constexpr std::array<const char *, 1> deviceExtensions {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    constexpr std::array<vk::PhysicalDeviceType, 3> suitableDeviceTypesInPriority {
            vk::PhysicalDeviceType::eDiscreteGpu,
            vk::PhysicalDeviceType::eIntegratedGpu,
            vk::PhysicalDeviceType::eVirtualGpu
    };

    void checkExtensions(const std::vector<const char *> &requiredExtensions) {
        const auto availableExtensions = vk::enumerateInstanceExtensionProperties();
        std::unordered_set<std::string> availableExtensionNames;
        availableExtensionNames.reserve(availableExtensions.size());
        for (const auto &availableExtension : availableExtensions) {
            availableExtensionNames.emplace(availableExtension.extensionName);
        }
        for (const auto requiredExtension : requiredExtensions) {
            if (availableExtensionNames.find(requiredExtension) == availableExtensionNames.cend()) {
                throw vk::ExtensionNotPresentError(requiredExtension);
            }
        }
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice &device) {
        const auto availableExtensions = device.enumerateDeviceExtensionProperties();
        std::unordered_set<std::string> availableExtensionNames;
        availableExtensionNames.reserve(availableExtensions.size());
        for (const auto &availableExtension : availableExtensions) {
            availableExtensionNames.emplace(availableExtension.extensionName);
        }

        return std::all_of(
                deviceExtensions.cbegin(), deviceExtensions.cend(),
                [&availableExtensionNames](const auto &requiredExtension) {
                    return availableExtensionNames.find(requiredExtension) != availableExtensionNames.cend();
                });
    }

    vk::UniqueInstance createInstance() {
        debug::checkValidationLayerSupport();

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
                debug::validationLayersSize(),
                debug::validationLayersData(),
                requiredExtensions.size(),
                requiredExtensions.data()
        );

        debug::appendDebugInfo(createInfo);

        return vk::createInstanceUnique(createInfo);
    }
}

BaseGraphics::BaseGraphics()
        : startTime(std::chrono::high_resolution_clock::now()),
          window(framebufferResizeCallback, this),
          instance(createInstance()),
          dl{},
          dldi(*instance, getVkGetInstanceProcAddr()),
          debugMessenger(debug::setupDebugMessenger(*instance, dldi)),
          surface(window.createSurfaceUnique(*instance)),
          physicalDevice(pickPhysicalDevice()),
          queueFamilyIndices(findQueueFamilies()),
          logicalDevice(createLogicalDevice()),
          graphicsQueue(logicalDevice->getQueue(*queueFamilyIndices.graphicsFamily, 0)),
          presentQueue(logicalDevice->getQueue(*queueFamilyIndices.presentFamily, 0)),
          transferQueue(logicalDevice->getQueue(*queueFamilyIndices.transferFamily, 0)),
          descriptorSetLayout(createDescriptorSetLayout()),
          commandPool(createCommandPool(queueFamilyIndices.graphicsFamily, {})),
          transferCommandPool(createCommandPool(queueFamilyIndices.transferFamily,
                                                vk::CommandPoolCreateFlagBits::eResetCommandBuffer)),
          transferCommandBuffer(createTransferCommandBuffer()),
          vertexBuffer(createVertexBuffer()),
          indexBuffer(createIndexBuffer()),
          textureImage(createTextureImage()),
          textureImageView(createImageView(*textureImage.image, vk::Format::eR8G8B8A8Srgb)),
          textureSampler(createTextureSampler()),
          imageAvailableSemaphore({logicalDevice->createSemaphoreUnique({}),
                                   logicalDevice->createSemaphoreUnique({})}),
          renderFinishedSemaphore({logicalDevice->createSemaphoreUnique({}),
                                   logicalDevice->createSemaphoreUnique({})}),
          inFlightFences({createFence(),
                          createFence()}),
          res(*this),
          currentFrame(0) {
    std::cout << print_time << "Initialized" << std::endl;
}

SwapChainSupportDetails BaseGraphics::querySwapChainSupport(const vk::PhysicalDevice &requestedDevice) const {
    return {
            requestedDevice.getSurfaceCapabilitiesKHR(*surface),
            requestedDevice.getSurfaceFormatsKHR(*surface),
            requestedDevice.getSurfacePresentModesKHR(*surface)
    };
}

QueueFamilyIndices BaseGraphics::findQueueFamilies(const vk::PhysicalDevice &requestedDevice) const {
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
        && (queueFamilies[*result.graphicsFamily].queueFlags & vk::QueueFlagBits::eTransfer)) {
        result.transferFamily = result.graphicsFamily;
    }
    return result;
}

bool BaseGraphics::isDeviceSuitable(const vk::PhysicalDevice &requestedDevice,
                                    vk::PhysicalDeviceType desiredDeviceType) const {
    if (requestedDevice.getProperties().deviceType != desiredDeviceType) {
        return false;
    }
    const auto features = requestedDevice.getFeatures();
    if (!features.geometryShader) {
        return false;
    }
    if (!features.samplerAnisotropy) {
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

PFN_vkGetInstanceProcAddr BaseGraphics::getVkGetInstanceProcAddr() const {
    return dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
}

void BaseGraphics::run() {
    window.mainLoop([this]{
        if (res.drawFrame() == AfterDrawAction::RecreateSwapChain) {
            recreateSwapChain();
        }
    });
    logicalDevice->waitIdle();
}

vk::PhysicalDevice BaseGraphics::pickPhysicalDevice() const {
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

BaseGraphics::~BaseGraphics() {
    std::cout << print_time << "Finalization" << std::endl;
}

vk::UniqueDevice BaseGraphics::createLogicalDevice() const {
    float queuePriority = 1.0f;

    std::unordered_set<uint32_t> uniqueQueueFamilies{
            *queueFamilyIndices.graphicsFamily,
            *queueFamilyIndices.presentFamily,
            *queueFamilyIndices.transferFamily
    };

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
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
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    return physicalDevice.createDeviceUnique(vk::DeviceCreateInfo(
            {},
            queueCreateInfos.size(),
            queueCreateInfos.data(),
            debug::validationLayersSize(),
            debug::validationLayersData(),
            deviceExtensions.size(),
            deviceExtensions.data(),
            &deviceFeatures
    ));
}

vk::UniqueCommandPool BaseGraphics::createCommandPool(std::optional<uint32_t> queueFamily,
                                                      vk::CommandPoolCreateFlags commandPoolCreateFlags) const {
    return logicalDevice->createCommandPoolUnique(vk::CommandPoolCreateInfo(
            commandPoolCreateFlags,
            *queueFamily
    ));
}

void BaseGraphics::recreateSwapChain() {
    while(true) {
        const auto extent = window.getFramebufferSize();
        if (extent.width != 0 && extent.height != 0) {
            break;
        }
        GLFWWindow::waitEvents();
    }

    logicalDevice->waitIdle();

    res = SizeDependentResources(*this);
}

void BaseGraphics::framebufferResizeCallback(void *userPointer, int /*width*/, int /*height*/) {
    auto *_this = static_cast<BaseGraphics *>(userPointer);
    _this->res.framebufferResized = true;
}

uint32_t BaseGraphics::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    const auto memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

BufferWithMemory BaseGraphics::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                            vk::MemoryPropertyFlags properties) const {
    auto buffer = logicalDevice->createBufferUnique(vk::BufferCreateInfo{
            {},
            size,
            usage,
            vk::SharingMode::eExclusive
    });

    const auto memRequirements = logicalDevice->getBufferMemoryRequirements(*buffer);

    auto vertexBufferMemory = logicalDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
            memRequirements.size,
            findMemoryType(memRequirements.memoryTypeBits, properties)
    ));

    logicalDevice->bindBufferMemory(*buffer, *vertexBufferMemory, 0);

    return {
            std::move(vertexBufferMemory),
            std::move(buffer)
    };
}

BufferWithMemory BaseGraphics::createVertexBuffer() const {
    auto result = createBuffer(
            so::verticesSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyViaStagingBuffer(so::vertices.data(), so::verticesSize, result);
    return result;
}

BufferWithMemory BaseGraphics::createIndexBuffer() const {
    auto result = createBuffer(
            so::indicesSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyViaStagingBuffer(so::indices.data(), so::indicesSize, result);
    return result;
}

//TODO collect commands and async submit
template<typename CopyCommand, typename FlushBuffer>
void BaseGraphics::singleTimeCommand(CopyCommand copyCommand, FlushBuffer flushBuffer) const {
    transferCommandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    copyCommand(*transferCommandBuffer);
    transferCommandBuffer->end();
    flushBuffer();
    transferQueue.submit({
                                 vk::SubmitInfo({}, {}, {}, 1, &*transferCommandBuffer)
                         }, {});
    transferQueue.waitIdle();
    transferCommandBuffer->reset({});
}

vk::UniqueDescriptorSetLayout BaseGraphics::createDescriptorSetLayout() const {
    return logicalDevice->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(
            {},
            so::layoutBindings.size(),
            so::layoutBindings.data()
    ));
}

template<typename CopyCommandFactory>
void BaseGraphics::copyViaStagingBuffer(const void *src, size_t size, CopyCommandFactory copyCommandFactory) const {
    auto stagingBuffer = createBuffer(
            size, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached
    );

    void *data = logicalDevice->mapMemory(*stagingBuffer.bufferMemory, 0, size);
    std::memcpy(data, src, size);

    singleTimeCommand(copyCommandFactory(stagingBuffer), [this, &stagingBuffer] {
        logicalDevice->flushMappedMemoryRanges({vk::MappedMemoryRange(*stagingBuffer.bufferMemory, 0, VK_WHOLE_SIZE)});
    });
    logicalDevice->unmapMemory(*stagingBuffer.bufferMemory);
}

void BaseGraphics::copyViaStagingBuffer(const void *src, size_t size, const BufferWithMemory &dst) const {
    copyViaStagingBuffer(src, size, [size, &dst](const auto &stagingBuffer) {
        return [&stagingBuffer, &dst, size](const auto &c) {
            c.copyBuffer(*stagingBuffer.buffer, *dst.buffer, {vk::BufferCopy(0, 0, size)});
        };
    });
}

vk::UniqueCommandBuffer BaseGraphics::createTransferCommandBuffer() const {
    auto transferCommandBuffers = logicalDevice->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo(
                    *transferCommandPool,
                    vk::CommandBufferLevel::ePrimary,
                    1
            )
    );

    return std::move(transferCommandBuffers.front());
}

ImageWithMemory BaseGraphics::createTextureImage() const {
    Image image("textures/texture.jpg");

    auto textureImage_ = createImage(
            image.texWidth,
            image.texHeight,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    transitionImageLayout(*textureImage_.image, SwitchLayout{
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal});

    vk::BufferImageCopy region(
            0,
            0,
            0,
            vk::ImageSubresourceLayers(
                    vk::ImageAspectFlagBits::eColor,
                    0,
                    0,
                    1
            ),
            vk::Offset3D(0, 0, 0),
            vk::Extent3D(image.texWidth, image.texHeight, 1)
    );

    copyViaStagingBuffer(image.pixels, image.imageSize,
                         [&region, &textureImage_](const BufferWithMemory &stagingBuffer){
        return [&region, &stagingBuffer, &textureImage_](const vk::CommandBuffer &c){
            c.copyBufferToImage(*stagingBuffer.buffer, *textureImage_.image,
                                vk::ImageLayout::eTransferDstOptimal,{region});
        };
    });

    transitionImageLayout(*textureImage_.image, SwitchLayout{
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal});

    return textureImage_;
}

ImageWithMemory BaseGraphics::createImage(uint32_t width, uint32_t height, vk::Format format,
                                          vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                                          vk::MemoryPropertyFlags properties) const {
    auto textureImage_ = logicalDevice->createImageUnique(vk::ImageCreateInfo(
            {},
            vk::ImageType::e2D,
            format,
            vk::Extent3D(width, height, 1),
            1,
            1,
            vk::SampleCountFlagBits::e1,
            tiling,
            usage,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            vk::ImageLayout::eUndefined
    ));

    vk::MemoryRequirements memRequirements = logicalDevice->getImageMemoryRequirements(*textureImage_);

    auto textureImageMemory = logicalDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
            memRequirements.size,
            findMemoryType(memRequirements.memoryTypeBits, properties)
    ));

    logicalDevice->bindImageMemory(*textureImage_, *textureImageMemory, 0);

    return {
            std::move(textureImageMemory),
            std::move(textureImage_)
    };
}

void BaseGraphics::transitionImageLayout(vk::Image image, SwitchLayout switchLayout) const {
    vk::AccessFlags srcAccessMask;
    vk::AccessFlags dstAccessMask;
    vk::PipelineStageFlags srcStage;
    vk::PipelineStageFlags dstStage;

    if (switchLayout == SwitchLayout{
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal}) {
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        dstStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (switchLayout == SwitchLayout{
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal}) {
        srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        dstAccessMask = vk::AccessFlagBits::eShaderRead;
        srcStage = vk::PipelineStageFlagBits::eTransfer;
        dstStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vk::ImageMemoryBarrier barrier(
            srcAccessMask,
            dstAccessMask,
            switchLayout.oldLayout,
            switchLayout.newLayout,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            image,
            vk::ImageSubresourceRange(
                    vk::ImageAspectFlagBits::eColor,
                    0,
                    1,
                    0,
                    1
            )
    );
    singleTimeCommand([&barrier, srcStage, dstStage](const vk::CommandBuffer &t) {
        t.pipelineBarrier(
                srcStage, dstStage,
                {},
                {},
                {},
                {barrier}
        );
    }, [] {});
}

vk::UniqueImageView BaseGraphics::createImageView(vk::Image image, vk::Format format) const {
    return logicalDevice->createImageViewUnique(vk::ImageViewCreateInfo(
            {},
            image,
            vk::ImageViewType::e2D,
            format,
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
    ));
}

vk::UniqueSampler BaseGraphics::createTextureSampler() const {
    return logicalDevice->createSamplerUnique(vk::SamplerCreateInfo(
            {},
            vk::Filter::eLinear,
            vk::Filter::eLinear,
            vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat,
            0.0f,
            VK_TRUE,
            16.0f,
            VK_FALSE,
            vk::CompareOp::eAlways,
            0.0f,
            0.0f,
            vk::BorderColor::eIntOpaqueBlack,
            VK_FALSE
    ));
}
