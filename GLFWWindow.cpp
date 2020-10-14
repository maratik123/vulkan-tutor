#include "GLFWWindow.h"

#include "debug.h"

void GLFWWindow::framebufferResizeCallback(GLFWwindow *window, int width, int height) {
    auto *_this = static_cast<GLFWWindow *>(glfwGetWindowUserPointer(window));
    (*_this->framebufferResizeCallbackFn)(_this->userPointer, width, height);
}

GLFWWindow::GLFWWindow(GLFWWindow::GLFWframebuffersizefun framebufferResizeCallbackFn, void *userPointer) :
    userPointer(userPointer),
    framebufferResizeCallbackFn(framebufferResizeCallbackFn) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    glfwSetWindowUserPointer(window, this);

    if (framebufferResizeCallbackFn) {
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }
}

GLFWWindow::~GLFWWindow() {
    if (framebufferResizeCallbackFn) {
        glfwSetFramebufferSizeCallback(window, nullptr);
    }
    glfwSetWindowUserPointer(window, nullptr);
    glfwDestroyWindow(window);
}


std::vector<const char *> GLFWWindow::requiredExtensions() {
    uint32_t glfwExtensionCount;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> result(glfwExtensions, glfwExtensions + glfwExtensionCount);
    debug::appendDebugExtension(result);

    return result;
}

vk::Extent2D GLFWWindow::getFramebufferSize() const {
    int width;
    int height;
    glfwGetFramebufferSize(window, &width, &height);
    return {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
    };
}
