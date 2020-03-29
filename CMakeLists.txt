cmake_minimum_required(VERSION 3.15)
project(SASNN)

set(CMAKE_CXX_STANDARD 17)
if (DEBUG_LEVEL)
    add_definitions(-DDEBUG_LEVEL=${DEBUG_LEVEL})
endif()

add_executable(SASNN src/main.cpp src/Matrix2D.cpp include/Matrix2D.h include/stdafx.h src/Dataset.cpp include/Dataset.h include/functors.h src/functors.cpp)