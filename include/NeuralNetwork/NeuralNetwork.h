#pragma once


#include "../stdafx.h"

class NeuralNetwork {
public:
    virtual f32 GetAccuracy() = 0;
    virtual void Train(u64 steps) = 0;
    virtual void ModifyParam(char param_name, f32 value) = 0;
    virtual ~NeuralNetwork() = default;
};



