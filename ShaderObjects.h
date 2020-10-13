#ifndef VULKAN_TUTOR_SHADEROBJECTS_H
#define VULKAN_TUTOR_SHADEROBJECTS_H

#include <array>

#include "GLFWInclude.h"

#include "GLMInclude.h"

namespace so {
    struct Vertex {
        glm::vec2 pos{};
        glm::vec3 color{};

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

    struct UnifiedBufferObject {
        [[maybe_unused]] alignas(16) glm::mat4 model{};
        [[maybe_unused]] alignas(16) glm::mat4 view{};
        [[maybe_unused]] alignas(16) glm::mat4 proj{};

        static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding = vk::DescriptorSetLayoutBinding(
                0,
                vk::DescriptorType::eUniformBuffer,
                1,
                vk::ShaderStageFlagBits::eVertex
        );
    };

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
}
#endif //VULKAN_TUTOR_SHADEROBJECTS_H
