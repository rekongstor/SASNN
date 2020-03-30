cmake_minimum_required(VERSION 3.15)
project(SASNN)

set(CMAKE_CXX_STANDARD 17)
if (DEBUG_LEVEL)
    add_definitions(-DDEBUG_LEVEL=${DEBUG_LEVEL})
endif()

add_executable(SASNN src/main.cpp src/Core/Matrix2D.cpp include/Core/Matrix2D.h include/stdafx.h src/Dataset/Dataset.cpp include/Dataset/Dataset.h src/Core/functors.cpp include/Layer/Abstract/Layer.h src/Layer/LayerData.cpp src/Layer/LayerData.h src/Layer/LayerWeights.cpp src/Layer/LayerWeights.h src/Core/Random.cpp include/Core/Random.h src/NeuralNetwork/NeuralNetwork.cpp include/NeuralNetwork/NeuralNetwork.h src/Layer/LayerFullyConnected.cpp src/Layer/LayerFullyConnected.h src/Layer/Abstract/Layer.cpp src/Layer/Abstract/LayerDynamic.cpp include/Layer/Abstract/LayerDynamic.h src/Layer/LayerReLU.cpp src/Layer/LayerReLU.h src/Layer/LayerLeakyReLU.cpp src/Layer/LayerLeakyReLU.h src/Layer/LayerSigmoid.cpp src/Layer/LayerSigmoid.h src/Layer/LayerTanh.cpp src/Layer/LayerTanh.h src/Layer/LayerSoftMaxRow.cpp src/Layer/LayerSoftMaxRow.h)