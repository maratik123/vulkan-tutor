#include "GLFWResource.h"

#include <stdexcept>

GLFWResource::GLFWResource() {
    if (!glfwInit()) {
        throw std::runtime_error("Can not initialize GLFW library");
    }
}
