set(TARGET inference)

add_executable(${TARGET}
    main.cpp
    hf.cpp
    hf.h
)

target_include_directories(${TARGET} PRIVATE
    "${CMAKE_SOURCE_DIR}/thirdparty/tiny-cuda-nn/dependencies"
)

target_link_libraries(${TARGET} cxxopts stbi)