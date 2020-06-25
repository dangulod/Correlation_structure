cmake_minimum_required(VERSION 3.5)

project(EC_equation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("src")
link_directories("/usr/local/lib")

file(GLOB MODEL_SOURCES
    "src/Model/*.cpp")

add_executable(EC_equation main.cpp ${MODEL_SOURCES})
target_link_libraries(${PROJECT_NAME} armadillo nlopt)
