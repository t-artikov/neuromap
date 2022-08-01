set(TARGET trainer)
add_executable(${TARGET}
    Accuracy.cu
    Accuracy.h
    CombinedTilesSampler.cu
    CombinedTilesSampler.h
    Common.h
    DetailedTilesSampler.cu
    DetailedTilesSampler.h
    EmptyTilesSampler.cu
    EmptyTilesSampler.h
    HostMemory.h
    Model.cu
    Model.h
    SoftmaxCrossEntropyLoss.h
    ThreadSafeQueue.h
    main.cu
)

target_link_libraries(${TARGET} PRIVATE ${CUDA_LIBRARIES} tiny-cuda-nn cxxopts)
target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
