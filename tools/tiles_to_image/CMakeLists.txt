set(TARGET tiles_to_image)
add_executable(${TARGET} src/main.cpp)
target_link_libraries(${TARGET} cxxopts stbi)
