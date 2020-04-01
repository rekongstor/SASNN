#pragma once


#include "../stdafx.h"

class NeuralNetwork {
public:
    virtual f32 Test() = 0;
    virtual std::pair<f32, f32> Train(u64 steps) = 0;
    virtual void ModifyParam(char param_name, f32 value) = 0;
    virtual void Serialize(const char* filename) = 0;
    virtual ~NeuralNetwork() = default;
};



