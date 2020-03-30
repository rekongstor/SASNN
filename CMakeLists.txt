cmake_minimum_required(VERSION 3.15)
project(SASNN)

set(CMAKE_CXX_STANDARD 17)
if (DEBUG_LEVEL)
    add_definitions(-DDEBUG_LEVEL=${DEBUG_LEVEL})
endif()

add_executable(SASNN src/main.cpp src/Core/Matrix2D.cpp src/Core/Matrix2D.h include/stdafx.h src/Dataset/Dataset.cpp include/Dataset/Dataset.h src/Core/functors.cpp include/Layer/Abstract/Layer.h src/Layer/Simple/LayerData.cpp include/Layer/Simple/LayerData.h src/Layer/Simple/LayerWeights.cpp include/Layer/Simple/LayerWeights.h src/Core/Random.cpp include/Core/Random.h src/NeuralNetwork/NeuralNetwork.cpp include/NeuralNetwork/NeuralNetwork.h src/Layer/Functional/LayerFullyConnected.cpp include/Layer/Functional/LayerFullyConnected.h src/Layer/Abstract/Layer.cpp src/Layer/Abstract/LayerDynamic.cpp include/Layer/Abstract/LayerDynamic.h src/Layer/NLF/LayerReLU.cpp include/Layer/NLF/LayerReLU.h src/Layer/NLF/LayerLeakyReLU.cpp include/Layer/NLF/LayerLeakyReLU.h src/Layer/NLF/LayerSigmoid.cpp include/Layer/NLF/LayerSigmoid.h src/Layer/NLF/LayerTanh.cpp include/Layer/NLF/LayerTanh.h src/Layer/Functional/LayerExp.cpp include/Layer/Functional/LayerExp.h src/Layer/Functional/LayerStableSoftMax.cpp src/Layer/Functional/LayerStableSoftMax.h)