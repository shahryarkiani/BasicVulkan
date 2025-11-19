cmake_minimum_required(VERSION 3.8)

project("VulkanBasics")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

# GLFW Setup, we don't need tests, docs, examples
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("extern/glfw")

if(WIN32)
    set(COMPILE_SHADERS_CMD ${CMAKE_SOURCE_DIR}/compileShaders.bat)
else()
    set(COMPILE_SHADERS_CMD bash ${CMAKE_SOURCE_DIR}/compileShaders.sh)
endif()

# Compile and move shaders to output directory as well
add_custom_target(copy_shaders ALL
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND ${COMPILE_SHADERS_CMD}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/textures
        ${CMAKE_BINARY_DIR}/textures
        COMMENT "Compiling and copying shaders + textures to output directory"
)

add_executable(VulkanBasics "main.cpp")
include_directories(VulkanBasics PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_dependencies(VulkanBasics copy_shaders)

target_link_libraries(VulkanBasics PRIVATE Vulkan::Vulkan)
target_link_libraries(VulkanBasics PRIVATE glfw)