cmake_minimum_required(VERSION 3.1)

project(lodepng)

include_directories(include)

add_library(lodepng STATIC src/lodepng.cpp)

install(TARGETS lodepng ARCHIVE DESTINATION lib)
install(DIRECTORY include/lodepng DESTINATION include)