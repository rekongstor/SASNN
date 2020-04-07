#pragma once


#include "../../Core/Matrix2D.h"

class DecoratorGradientDescent {
public:
    virtual void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) = 0;
    virtual ~DecoratorGradientDescent() = default;
};



