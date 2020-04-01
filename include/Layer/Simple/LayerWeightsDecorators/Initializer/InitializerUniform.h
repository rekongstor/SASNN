#pragma once


#include "../../../../../src/Layer/Simple/LayerWeightsDecorators/DecoratorInitializer.h"

class InitializerUniform : public DecoratorInitializer {
    f32 a, b;
    void Initialize(Matrix2D &weights) override;
public:
    InitializerUniform(f32 a, f32 b);
};



