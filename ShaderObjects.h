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

        constexpr bool operator ==(const Vertex &other) const {
            return pos == other.pos
                   && color == other.color
                   && texCoord == other.texCoord;
        }

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

    constexpr std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings {
            UnifiedBufferObject::uboLayoutBinding, samplerLayoutBinding
    };
}

namespace std {
    template<>
    struct hash<so::Vertex> {
        size_t operator()(const so::Vertex &vertex) const {
            size_t seed = 0;
            seed = hash_combine(seed, hash<decltype(vertex.pos)>()(vertex.pos));
            seed = hash_combine(seed, hash<decltype(vertex.color)>()(vertex.color));
            seed = hash_combine(seed, hash<decltype(vertex.texCoord)>()(vertex.texCoord));
            return seed;
        }

    private:
        static constexpr size_t hash_combine(size_t seed, size_t hash) {
            hash += 0x9e3779b9 + (seed << 6u) + (seed >> 2u);
            return seed ^ hash;
        }
    };
}
#endif //VULKAN_TUTOR_SHADEROBJECTS_H
