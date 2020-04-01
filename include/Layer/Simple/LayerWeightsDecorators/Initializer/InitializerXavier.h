#pragma once


#include "../../../../../src/Layer/Simple/LayerWeightsDecorators/DecoratorInitializer.h"

class InitializerXavier : public DecoratorInitializer {
    f32 inputs;
    void Initialize(Matrix2D &weights) override;
public:
    InitializerXavier(f32 inputs);
};



