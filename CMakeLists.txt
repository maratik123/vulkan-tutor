cmake_minimum_required(VERSION 3.17)
project(vulkan_tutor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
#set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
#set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CMAKE_COMMAND} -E time")
#set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
     -Wall -Wextra -Wcast-qual -Wconversion-null -Wmissing-declarations \
     -Woverlength-strings -Wpointer-arith -Wunused-local-typedefs -Wunused-result \
     -Wvarargs -Wvla -Wwrite-strings")
endif()

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

function(add_asset TARGET ASSET)
    set(current-asset-path ${CMAKE_CURRENT_SOURCE_DIR}/${ASSET})
    set(current-output-path ${CMAKE_BINARY_DIR}/${ASSET})
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})
    add_custom_command(
            OUTPUT ${current-output-path}
            COMMAND ${CMAKE_COMMAND} -E copy ${current-asset-path} ${current-output-path}
            DEPENDS ${current-asset-path}
            IMPLICIT_DEPENDS CXX ${current-asset-path})
    set_source_files_properties(${current-output-path} PROPERTIES
            GENERATED TRUE
            HEADER_FILE_ONLY TRUE)
    target_sources(${TARGET} PRIVATE ${current-output-path})
endfunction(add_asset)

function(add_texture TARGET TEXTURE)
    add_asset(${TARGET} textures/${TEXTURE})
endfunction(add_texture)

function(add_model TARGET MODEL)
    add_asset(${TARGET} models/${MODEL})
endfunction(add_model)

add_executable(vulkan_tutor
        Model.cpp Model.h
        BaseGraphics.cpp BaseGraphics.h
        SizeDependentResources.cpp SizeDependentResources.h
        Image.cpp Image.h
        Debug.cpp Debug.h
        Main.cpp
        GLFWWindow.cpp GLFWWindow.h
        GLFWResource.cpp GLFWResource.h
        Utils.cpp Utils.h
        ShaderObjects.h
        GLFWInclude.h
        GLMInclude.h
        STBInclude.h
        Common.h
        TOLInclude.h)

add_shader(vulkan_tutor shader.vert)
add_shader(vulkan_tutor shader.frag)

add_texture(vulkan_tutor texture.jpg)
add_texture(vulkan_tutor viking_room.png)

add_model(vulkan_tutor viking_room.obj)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GLM REQUIRED glm)

target_include_directories(vulkan_tutor SYSTEM PRIVATE
        ${GLM_INCLUDE_DIRS}
        stb
        tinyobjloader)
target_link_libraries(vulkan_tutor
        glfw
        Vulkan::Vulkan
        ${CMAKE_DL_LIBS})
