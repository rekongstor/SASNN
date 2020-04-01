#pragma once


#include "../../../../include/Core/Matrix2D.h"

class DecoratorGradientDescent {
public:
    virtual void subGrad(Matrix2D &weights, const Matrix2D &grad) = 0;
};



