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

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice requestedDevice) {
        const auto availableExtensions = requestedDevice.enumerateDeviceExtensionProperties();
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

    constexpr bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }
}

BaseGraphics::BaseGraphics()
        : startTime(std::chrono::high_resolution_clock::now()),
          window(framebufferResizeCallback, &res),
          instance(createInstance()),
          dl(),
          dldi(*instance, getVkGetInstanceProcAddr()),
          debugMessenger(debug::setupDebugMessenger(*instance, dldi)),
          surface(window.createSurfaceUnique(*instance)),
          physicalDevice(pickPhysicalDevice()),
          msaaSamples(getMaxUsableSampleCount()),
          queueFamilyIndices(findQueueFamilies()),
          device(createLogicalDevice()),
          graphicsQueue(device->getQueue(*queueFamilyIndices.graphicsFamily, 0)),
          presentQueue(device->getQueue(*queueFamilyIndices.presentFamily, 0)),
          transferQueue(device->getQueue(*queueFamilyIndices.transferFamily, 0)),
          descriptorSetLayout(createDescriptorSetLayout()),
          graphicsCommandPool(createCommandPool(queueFamilyIndices.graphicsFamily, {})),
          transferCommandPool(createCommandPool(queueFamilyIndices.transferFamily, {})),
          modelBuffers(createModelBuffers()),
          textureImage(createTextureImage()),
          textureImageView(createImageView(*textureImage.image.image, vk::Format::eR8G8B8A8Srgb,
                                           vk::ImageAspectFlagBits::eColor, textureImage.mipLevels)),
          textureSampler(createTextureSampler()),
          imageAvailableSemaphore({device->createSemaphoreUnique({}),
                                   device->createSemaphoreUnique({})}),
          renderFinishedSemaphore({device->createSemaphoreUnique({}),
                                   device->createSemaphoreUnique({})}),
          inFlightFences({createFence(),
                          createFence()}),
          depthFormat(findDepthFormat()),
          vertShaderModule(createShaderModule(readFile("shaders/shader.vert.spv"))),
          fragShaderModule(createShaderModule(readFile("shaders/shader.frag.spv"))),
          res(*this),
          currentFrame(0) {
    std::cout << print_time << "Initialized" << std::endl;
}

SwapChainSupportDetails BaseGraphics::querySwapChainSupport(const vk::PhysicalDevice requestedDevice) const {
    return {
            requestedDevice.getSurfaceCapabilitiesKHR(*surface),
            requestedDevice.getSurfaceFormatsKHR(*surface),
            requestedDevice.getSurfacePresentModesKHR(*surface)
    };
}

QueueFamilyIndices BaseGraphics::findQueueFamilies(const vk::PhysicalDevice requestedDevice) const {
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

bool BaseGraphics::isDeviceSuitable(const vk::PhysicalDevice requestedDevice,
                                    const vk::PhysicalDeviceType desiredDeviceType) const {
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
    device->waitIdle();
}

vk::PhysicalDevice BaseGraphics::pickPhysicalDevice() const {
    const auto devices = instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (auto deviceType : suitableDeviceTypesInPriority) {
        const auto it = std::find_if(
                devices.cbegin(), devices.cend(),
                [this, deviceType](auto device_) { return isDeviceSuitable(device_, deviceType); }
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
    deviceFeatures.sampleRateShading = VK_TRUE;

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
    return device->createCommandPoolUnique(vk::CommandPoolCreateInfo(
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

    device->waitIdle();

    res = SizeDependentResources(*this);
}

void BaseGraphics::framebufferResizeCallback(void *userPointer, int /*width*/, int /*height*/) {
    auto *res = static_cast<SizeDependentResources *>(userPointer);
    res->framebufferResized = true;
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
    auto buffer = device->createBufferUnique(vk::BufferCreateInfo{
            {},
            size,
            usage,
            vk::SharingMode::eExclusive
    });

    const auto memRequirements = device->getBufferMemoryRequirements(*buffer);

    auto vertexBufferMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(
            memRequirements.size,
            findMemoryType(memRequirements.memoryTypeBits, properties)
    ));

    device->bindBufferMemory(*buffer, *vertexBufferMemory, 0);

    return {
            std::move(vertexBufferMemory),
            std::move(buffer)
    };
}

BufferWithMemory BaseGraphics::createVertexBuffer(const Model &model) const {
    auto size = sizeof(decltype(model.vertices)::value_type) * model.vertices.size();
    auto result = createBuffer(
            size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyViaStagingBuffer(model.vertices.data(), size, result);
    return result;
}

BufferWithMemory BaseGraphics::createIndexBuffer(const Model &model) const {
    auto size = sizeof(decltype(model.indices)::value_type) * model.indices.size();
    auto result = createBuffer(
            size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyViaStagingBuffer(model.indices.data(), size, result);
    return result;
}

//TODO collect commands and async submit
template<typename CopyCommand, typename FlushBuffer>
void BaseGraphics::singleTimeCommand(vk::Queue queue, vk::CommandPool commandPool, CopyCommand copyCommand,
                                     FlushBuffer flushBuffer) const {
    auto commandBuffer = createCommandBuffer(commandPool);
    commandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    copyCommand(*commandBuffer);
    commandBuffer->end();
    flushBuffer();
    queue.submit({
                                 vk::SubmitInfo({}, {}, {}, 1, &*commandBuffer)
                         }, {});
    queue.waitIdle();
}

vk::UniqueDescriptorSetLayout BaseGraphics::createDescriptorSetLayout() const {
    return device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(
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

    void *data = device->mapMemory(*stagingBuffer.bufferMemory, 0, size);
    std::memcpy(data, src, size);

    singleTimeCommand(transferQueue, *transferCommandPool, copyCommandFactory(stagingBuffer), [this, &stagingBuffer] {
        device->flushMappedMemoryRanges({vk::MappedMemoryRange(*stagingBuffer.bufferMemory, 0, VK_WHOLE_SIZE)});
    });
    device->unmapMemory(*stagingBuffer.bufferMemory);
}

void BaseGraphics::copyViaStagingBuffer(const void *src, size_t size, const BufferWithMemory &dst) const {
    copyViaStagingBuffer(src, size, [size, &dst](const auto &stagingBuffer) {
        return [&stagingBuffer, &dst, size](auto c) {
            c.copyBuffer(*stagingBuffer.buffer, *dst.buffer, {vk::BufferCopy(0, 0, size)});
        };
    });
}

vk::UniqueCommandBuffer BaseGraphics::createCommandBuffer(vk::CommandPool commandPool) const {
    auto transferCommandBuffers = device->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo(
                    commandPool,
                    vk::CommandBufferLevel::ePrimary,
                    1
            )
    );

    return std::move(transferCommandBuffers.front());
}

TextureImage BaseGraphics::createTextureImage() const {
    Image image("textures/viking_room.png");
    uint32_t mipLevels = std::floor(std::log2(std::max(image.texWidth, image.texHeight))) + 1;

    auto textureImage_ = createImage(
            image.texWidth,
            image.texHeight,
            mipLevels,
            vk::SampleCountFlagBits::e1,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferSrc
            | vk::ImageUsageFlagBits::eTransferDst
            | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    transitionImageLayout(transferQueue, *transferCommandPool, *textureImage_.image, vk::Format::eR8G8B8A8Srgb,
                          SwitchLayout{
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferDstOptimal},
                          mipLevels);

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

    generateMipmaps(*textureImage_.image, vk::Format::eR8G8B8A8Srgb, image.texWidth, image.texHeight, mipLevels);

    return {
        std::move(textureImage_),
        mipLevels
    };
}

ImageWithMemory BaseGraphics::createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                                          vk::SampleCountFlagBits numSamples, vk::Format format,
                                          vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                                          vk::MemoryPropertyFlags properties) const {
    auto textureImage_ = device->createImageUnique(vk::ImageCreateInfo(
            {},
            vk::ImageType::e2D,
            format,
            vk::Extent3D(width, height, 1),
            mipLevels,
            1,
            numSamples,
            tiling,
            usage,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            vk::ImageLayout::eUndefined
    ));

    vk::MemoryRequirements memRequirements = device->getImageMemoryRequirements(*textureImage_);

    auto textureImageMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(
            memRequirements.size,
            findMemoryType(memRequirements.memoryTypeBits, properties)
    ));

    device->bindImageMemory(*textureImage_, *textureImageMemory, 0);

    return {
            std::move(textureImageMemory),
            std::move(textureImage_)
    };
}

void BaseGraphics::transitionImageLayout(vk::Queue queue, vk::CommandPool commandPool, vk::Image image,
                                         vk::Format format, SwitchLayout switchLayout, uint32_t mipLevels) const {
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
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal}) {
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
                        | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        dstStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
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

    vk::ImageAspectFlags aspectMask;

    if (switchLayout.newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        aspectMask = vk::ImageAspectFlagBits::eDepth;
        if (hasStencilComponent(format)) {
            aspectMask |= vk::ImageAspectFlagBits::eStencil;
        }
    } else {
        aspectMask = vk::ImageAspectFlagBits::eColor;
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
                    aspectMask,
                    0,
                    mipLevels,
                    0,
                    1
            )
    );
    singleTimeCommand(queue, commandPool, [&barrier, srcStage, dstStage](auto t) {
        t.pipelineBarrier(
                srcStage, dstStage,
                {},
                {},
                {},
                {barrier}
        );
    }, [] {});
}

vk::UniqueImageView BaseGraphics::createImageView(vk::Image image, vk::Format format,
                                                  vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const {
    return device->createImageViewUnique(vk::ImageViewCreateInfo(
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
                    aspectFlags,
                    0,
                    mipLevels,
                    0,
                    1
            }
    ));
}

vk::UniqueSampler BaseGraphics::createTextureSampler() const {
    return device->createSamplerUnique(vk::SamplerCreateInfo(
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
            static_cast<float>(textureImage.mipLevels),
            vk::BorderColor::eIntOpaqueBlack,
            VK_FALSE
    ));
}

vk::Format BaseGraphics::findSupportedFormat(vk::ArrayProxy<const vk::Format> candidates, vk::ImageTiling tiling,
                                             vk::FormatFeatureFlags features) const {
    switch (tiling) {
        case vk::ImageTiling::eLinear: {
            const auto it = std::find_if(candidates.begin(), candidates.end(), [this, features](auto format) {
                const auto props = physicalDevice.getFormatProperties(format);
                return (props.linearTilingFeatures & features) == features;
            });
            if (it != candidates.end()) {
                return *it;
            }
            break;
        }
        case vk::ImageTiling::eOptimal: {
            const auto it = std::find_if(candidates.begin(), candidates.end(), [this, features](auto format) {
                const auto props = physicalDevice.getFormatProperties(format);
                return (props.optimalTilingFeatures & features) == features;
            });
            if (it != candidates.end()) {
                return *it;
            }
            break;
        }
        default:
            break;
    }

    throw std::runtime_error("failed to find supported format!");
}

vk::UniqueShaderModule BaseGraphics::createShaderModule(const std::vector<char> &code) const {
    return device->createShaderModuleUnique(vk::ShaderModuleCreateInfo(
            {},
            code.size(),
            reinterpret_cast<const uint32_t *>(code.data())
    ));
}

ModelBuffers BaseGraphics::createModelBuffers() const {
    Model model("models/viking_room.obj");
    return ModelBuffers {
            createVertexBuffer(model),
            createIndexBuffer(model),
            model.indices.size()
    };
}

void BaseGraphics::generateMipmaps(vk::Image image, vk::Format imageFormat, uint32_t texWidth, uint32_t texHeight,
                                   uint32_t mipLevels) const {
    auto formatProperties = physicalDevice.getFormatProperties(imageFormat);
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        //TODO software mipmap generation
        throw std::runtime_error("texture image format does not support linear blitting!");
    }
    singleTimeCommand(graphicsQueue, *graphicsCommandPool,
                      [image, texWidth, texHeight, mipLevels](vk::CommandBuffer commandBuffer){
                          vk::ImageMemoryBarrier barrier(
                                  {},
                                  {},
                                  {},
                                  {},
                                  VK_QUEUE_FAMILY_IGNORED,
                                  VK_QUEUE_FAMILY_IGNORED,
                                  image,
                                  vk::ImageSubresourceRange(
                                          vk::ImageAspectFlagBits::eColor,
                                          {},
                                          1,
                                          0,
                                          1
                                  )
                          );

                          int32_t mipWidth = texWidth;
                          int32_t mipHeight = texHeight;

                          for (uint32_t i = 1; i < mipLevels; ++i) {
                              barrier.subresourceRange.baseMipLevel = i - 1;
                              barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
                              barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
                              barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
                              barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

                              commandBuffer.pipelineBarrier(
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      {},
                                      {},
                                      {},
                                      {barrier}
                              );

                              vk::ImageBlit blit(
                                      vk::ImageSubresourceLayers(
                                              vk::ImageAspectFlagBits::eColor,
                                              i - 1,
                                              0,
                                              1
                                      ),
                                      {{
                                               {0, 0, 0},
                                               {mipWidth, mipHeight, 1}
                                       }},
                                      vk::ImageSubresourceLayers(
                                              vk::ImageAspectFlagBits::eColor,
                                              i,
                                              0,
                                              1
                                      ),
                                      {{
                                               {0, 0, 0},
                                               {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1}
                                       }}
                              );
                              commandBuffer.blitImage(
                                      image, vk::ImageLayout::eTransferSrcOptimal,
                                      image, vk::ImageLayout::eTransferDstOptimal,
                                      {blit},
                                      vk::Filter::eLinear
                              );

                              barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
                              barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                              barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
                              barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

                              commandBuffer.pipelineBarrier(
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eFragmentShader,
                                      {},
                                      {},
                                      {},
                                      {barrier}
                              );
                              if (mipWidth > 1) {
                                  mipWidth /= 2;
                              }
                              if (mipHeight > 1) {
                                  mipHeight /= 2;
                              }
                          }

                          barrier.subresourceRange.baseMipLevel = mipLevels - 1;
                          barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
                          barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                          barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
                          barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

                          commandBuffer.pipelineBarrier(
                                  vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eFragmentShader,
                                  {},
                                  {},
                                  {},
                                  {barrier}
                          );
                      }, []{});
}

vk::SampleCountFlagBits BaseGraphics::getMaxUsableSampleCount() const {
    auto physicalDeviceProperties = physicalDevice.getProperties();

    vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts
            & physicalDeviceProperties.limits.framebufferDepthSampleCounts;

    if (counts & vk::SampleCountFlagBits::e64) {
        return vk::SampleCountFlagBits::e64;
    }
    if (counts & vk::SampleCountFlagBits::e32) {
        return vk::SampleCountFlagBits::e32;
    }
    if (counts & vk::SampleCountFlagBits::e16) {
        return vk::SampleCountFlagBits::e16;
    }
    if (counts & vk::SampleCountFlagBits::e8) {
        return vk::SampleCountFlagBits::e8;
    }
    if (counts & vk::SampleCountFlagBits::e4) {
        return vk::SampleCountFlagBits::e4;
    }
    if (counts & vk::SampleCountFlagBits::e2) {
        return vk::SampleCountFlagBits::e2;
    }
    return vk::SampleCountFlagBits::e1;
}
