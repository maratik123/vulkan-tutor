#include "GLFWResource.h"

#include <stdexcept>

#include "GLFWInclude.h"

GLFWResource::GLFWResource() {
    if (!glfwInit()) {
        throw std::runtime_error("Can not initialize GLFW library");
    }
}

GLFWResource::~GLFWResource() {
    glfwTerminate();
}
