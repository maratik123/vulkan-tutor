#include <iostream>

#include "GLFWResource.h"
#include "HelloTriangle.h"

namespace {
    void actualMain() {
        GLFWResource glfwResource{};

        HelloTriangle app{};

        app.run();
    }

    template<typename F>
    int wrap(F f) {
        try {
            f();
        } catch (const std::system_error& e) {
            std::cerr << print_time << "System error: " << e.what() << std::endl;
            if (exitWithStackTrace) {
                throw e;
            }
            return e.code().value();
        } catch (const std::exception& e) {
            std::cerr << print_time << "General failure: " << e.what() << std::endl;
            if (exitWithStackTrace) {
                throw e;
            }
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
}

int main() {
    std::cout << print_time << "Start app" << std::endl;
    int result = wrap(actualMain);
    std::cout << print_time << "Exit" << std::endl;
    return result;
}
