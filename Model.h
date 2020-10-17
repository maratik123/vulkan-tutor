#ifndef VULKAN_TUTOR_MODEL_H
#define VULKAN_TUTOR_MODEL_H

#include <vector>

#include "ShaderObjects.h"

class Model {
public:
    explicit Model(const char* fileName);
    std::vector<so::Vertex> vertices;
    std::vector<uint32_t> indices;
};


#endif //VULKAN_TUTOR_MODEL_H
