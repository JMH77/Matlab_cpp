cmake_minimum_required(VERSION 3.29)
project(PM_FFT)


set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(${PROJECT_NAME} main01.cpp main.h)


find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
#include_directories(${Eigen3_INCLUDE_DIRS})

find_package(SOCI CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SOCI::soci_core SOCI::soci_odbc)
