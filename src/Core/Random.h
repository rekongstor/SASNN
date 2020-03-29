#pragma once

#include <random>
#include "../../include/stdafx.h"

class Random {
protected:
    std::mt19937 engine;
    explicit Random(u64 seed);
public:
    virtual f32 Next() = 0;
};


class RandomUniform : public Random {
    std::uniform_real_distribution<f32> distribution;
public:
    explicit RandomUniform(f32 minRange = 0.f, f32 maxRange = 1.f, u64 seed = 0);
private:
    f32 Next() override;
};

class RandomGaussian : public Random {
    std::normal_distribution<f32> distribution;
public:
    explicit RandomGaussian(f32 mean = 0.f, f32 dev = 1.f, u64 seed = 0);
private:
    f32 Next() override;
};