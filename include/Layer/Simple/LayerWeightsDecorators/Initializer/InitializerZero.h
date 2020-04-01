#pragma once


#include "../DecoratorInitializer.h"

class InitializerZero : public DecoratorInitializer {
    void Initialize(Matrix2D &weights) override;
};



