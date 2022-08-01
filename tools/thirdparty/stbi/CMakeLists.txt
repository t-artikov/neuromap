set(TARGET stbi)
add_library(${TARGET} STATIC src/stbi.cpp)
target_include_directories(${TARGET} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
