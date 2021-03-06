#pragma once


#include "../stdafx.h"
#include "../Core/Matrix2D.h"

class NeuralNetwork {
public:
    virtual std::pair<f32, f32> Test() = 0;
    virtual std::pair<f32, f32> Train() = 0;
    virtual void Serialize(const char *filename) = 0;
    virtual void Deserialize(const char *filename) = 0;
    virtual void ModifyParam(char param_name, f32 value) = 0;
    virtual void Use() = 0;
    virtual ~NeuralNetwork() = default;
};



