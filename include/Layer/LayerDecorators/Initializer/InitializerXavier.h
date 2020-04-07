#pragma once


#include "../DecoratorInitializer.h"

class InitializerXavier : public DecoratorInitializer {
    f32 inputs;
    void Initialize(Matrix2D &weights) override;
public:
    explicit InitializerXavier(f32 inputs);
};



