#pragma once


#include "../DecoratorInitializer.h"

class InitializerUniform : public DecoratorInitializer {
    f32 a, b;
    void Initialize(Matrix2D &weights) override;
public:
    InitializerUniform(f32 a, f32 b);
};



