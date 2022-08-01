set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CUDA_NVCC_FLAGS 
    "-Xcompiler=-mf16c"
    "-Xcompiler=-Wno-float-conversion"
    "-Xcompiler=-fno-strict-aliasing"
    "--extended-lambda"
    "--expt-relaxed-constexpr"
)
