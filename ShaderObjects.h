#ifndef VULKAN_TUTOR_SHADEROBJECTS_H
#define VULKAN_TUTOR_SHADEROBJECTS_H

#include <array>

#include "GLFWInclude.h"

#include "GLMInclude.h"

namespace so {
    struct Vertex {
        glm::vec3 pos{};
        glm::vec3 color{};
        glm::vec2 texCoord{};

        static constexpr vk::VertexInputBindingDescription bindingDescription() {
            return {
                    0,
                    sizeof(Vertex),
                    vk::VertexInputRate::eVertex
            };
        }

        static constexpr std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
            return {{
                            vk::VertexInputAttributeDescription(
                                    0,
                                    0,
                                    vk::Format::eR32G32B32Sfloat,
                                    offsetof(Vertex, pos)
                            ),
                            vk::VertexInputAttributeDescription(
                                    1,
                                    0,
                                    vk::Format::eR32G32B32Sfloat,
                                    offsetof(Vertex, color)
                            ),
                            vk::VertexInputAttributeDescription(
                                    2,
                                    0,
                                    vk::Format::eR32G32Sfloat,
                                    offsetof(Vertex, texCoord)
                            )
                    }};
        }
    };

    struct UnifiedBufferObject {
        alignas(16) glm::mat4 model{};
        alignas(16) glm::mat4 view{};
        alignas(16) glm::mat4 proj{};

        static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding = vk::DescriptorSetLayoutBinding(
                0,
                vk::DescriptorType::eUniformBuffer,
                1,
                vk::ShaderStageFlagBits::eVertex
        );
    };

    constexpr vk::DescriptorSetLayoutBinding samplerLayoutBinding = vk::DescriptorSetLayoutBinding(
            1,
            vk::DescriptorType::eCombinedImageSampler,
            1,
            vk::ShaderStageFlagBits::eFragment
    );

    constexpr std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings = {
            UnifiedBufferObject::uboLayoutBinding, samplerLayoutBinding
    };

    constexpr std::array<Vertex, 4> vertices{{
                                                     {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                                                     {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                                                     {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                                                     {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
                                             }};
    constexpr std::array<uint16_t, 6> indices{{
                                                      0, 1, 2, 2, 3, 0
                                              }};

    constexpr size_t verticesSize = sizeof(decltype(vertices)::value_type) * vertices.size();
    constexpr size_t indicesSize = sizeof(decltype(indices)::value_type) * indices.size();
}
#endif //VULKAN_TUTOR_SHADEROBJECTS_H
