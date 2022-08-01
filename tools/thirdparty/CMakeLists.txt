set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(cxxopts)
add_subdirectory(stbi)

SET(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
SET(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(tiny-cuda-nn)
target_include_directories(tiny-cuda-nn PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/tiny-cuda-nn/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tiny-cuda-nn/dependencies"
)
target_compile_definitions(tiny-cuda-nn PUBLIC ${TCNN_DEFINITIONS})
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
