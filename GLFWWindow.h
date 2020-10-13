#ifndef VULKAN_TUTOR_GLFWWINDOW_H
#define VULKAN_TUTOR_GLFWWINDOW_H

#include <vector>
#include "GLFWInclude.h"

class GLFWWindow {
public:
    typedef void (* GLFWframebuffersizefun)(void *, int, int);

    explicit GLFWWindow(GLFWframebuffersizefun framebufferResizeCallback = nullptr, void *userPointer = nullptr);
    ~GLFWWindow();

    template<typename LoopFn>
    void mainLoop(LoopFn loopFn) const;
    [[nodiscard]] static std::vector<const char *> requiredExtensions();
    template<typename Dispatch = VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>
    [[nodiscard]]
    typename vk::ResultValueType<vk::UniqueHandle<vk::SurfaceKHR, Dispatch>>::type
    createSurfaceUnique(const vk::Instance &instance, vk::Optional<const vk::AllocationCallbacks> allocator = nullptr,
                              Dispatch const &d = VULKAN_HPP_DEFAULT_DISPATCHER) const;
    [[nodiscard]] vk::Extent2D getFramebufferSize() const;
    static void waitEvents() {
        glfwWaitEvents();
    }

    static constexpr uint32_t WIDTH = 800;
    static constexpr uint32_t HEIGHT = 600;

private:
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    GLFWwindow *window;
    void *userPointer;
    GLFWframebuffersizefun framebufferResizeCallbackFn;
};

template<typename Dispatch>
VULKAN_HPP_INLINE typename vk::ResultValueType<vk::UniqueHandle<vk::SurfaceKHR, Dispatch>>::type
GLFWWindow::createSurfaceUnique(const vk::Instance &instance, vk::Optional<const vk::AllocationCallbacks> allocator,
                          Dispatch const &d) const {
    vk::SurfaceKHR surface;
    auto result = static_cast<vk::Result>(glfwCreateWindowSurface(
            instance, window,
            reinterpret_cast<const VkAllocationCallbacks *>(static_cast<const vk::AllocationCallbacks *>(allocator)),
            reinterpret_cast<VkSurfaceKHR *>(&surface)
    ));

    vk::ObjectDestroy<vk::Instance, Dispatch> deleter(instance, allocator, d);
    return vk::createResultValue<vk::SurfaceKHR, Dispatch>(result, surface, "GLFWWindow::createSurface", deleter);
}

template<typename LoopFn>
void GLFWWindow::mainLoop(LoopFn loopFn) const {
    while (!glfwWindowShouldClose(window)) {
        loopFn();
        glfwPollEvents();
    }
}

#endif //VULKAN_TUTOR_GLFWWINDOW_H
