cmake_minimum_required(VERSION 3.17)
project(vulkan_tutor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_subdirectory(abseil-cpp)

function(add_shader TARGET SHADER)
    find_program(GLSLC glslc)
    set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER})
    set(current-output-path ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv)
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})
    add_custom_command(
            OUTPUT ${current-output-path}
            COMMAND ${GLSLC} -O -o ${current-output-path} ${current-shader-path}
            DEPENDS ${current-shader-path}
            IMPLICIT_DEPENDS CXX ${current-shader-path}
            VERBATIM)
    set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${current-output-path})
endfunction(add_shader)

add_executable(vulkan_tutor
        main.cpp
        GLFWInclude.h
        HelloTriangle.cpp HelloTriangle.h
        GLFWWindow.cpp GLFWWindow.h
        GLFWResource.cpp GLFWResource.h
        utils.cpp utils.h)

add_shader(vulkan_tutor shader.vert)
add_shader(vulkan_tutor shader.frag)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GLM REQUIRED glm)

target_include_directories(vulkan_tutor PRIVATE
        ${GLM_INCLUDE_DIRS}
        stb)
target_link_libraries(vulkan_tutor
        glfw
        Vulkan::Vulkan
        ${CMAKE_DL_LIBS}
        absl::flat_hash_set
        absl::time)
