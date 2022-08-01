cmake_minimum_required(VERSION 3.18)
project(NeuroMapTools
    LANGUAGES CXX CUDA
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include("cmake/cuda.cmake")

add_subdirectory(thirdparty)
add_subdirectory(tiles_to_image)
add_subdirectory(trainer)
add_subdirectory(inference)
