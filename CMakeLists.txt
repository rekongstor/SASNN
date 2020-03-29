cmake_minimum_required(VERSION 3.15)
project(SASNN)

set(CMAKE_CXX_STANDARD 17)
if (DEBUG_LEVEL)
    add_definitions(-DDEBUG_LEVEL=${DEBUG_LEVEL})
endif()

add_executable(SASNN src/main.cpp src/Core/Matrix2D.cpp include/Core/Matrix2D.h include/stdafx.h src/Dataset/Dataset.cpp include/Dataset/Dataset.h include/Core/functors.h src/Core/functors.cpp include/Layer/Layer.h src/Layer/LayerData.cpp include/Layer/LayerData.h src/Layer/LayerWeights.cpp include/Layer/LayerWeights.h src/Core/Random.cpp include/Core/Random.h)