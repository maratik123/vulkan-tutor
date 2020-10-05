#ifndef VULKAN_TUTOR_GLFWRESOURCE_H
#define VULKAN_TUTOR_GLFWRESOURCE_H

#include "GLFWInclude.h"

class GLFWResource {
public:
    GLFWResource();
    ~GLFWResource() { glfwTerminate(); }
};

#endif //VULKAN_TUTOR_GLFWRESOURCE_H
