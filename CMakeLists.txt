cmake_minimum_required(VERSION 3.8)

project("VulkTutorial")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

# GLFW Setup, we don't need tests, docs, examples
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("extern/glfw")

# Compile and move shaders to output directory as well
add_custom_target(copy_shaders ALL
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND ./compileShaders.bat
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
        COMMENT "Compiling and copying shaders to output directory"
)

add_executable(VulkTutorial "main.cpp")
add_dependencies(VulkTutorial copy_shaders)

target_link_libraries(VulkTutorial PRIVATE Vulkan::Vulkan)
target_link_libraries(VulkTutorial PRIVATE glfw)