#pragma once


#include "../../../../../src/Layer/Simple/LayerWeightsDecorators/DecoratorInitializer.h"

class InitializerZero : public DecoratorInitializer {
    void Initialize(Matrix2D &weights) override;
};



