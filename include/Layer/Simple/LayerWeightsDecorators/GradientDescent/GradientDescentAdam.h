#pragma once


#include "../DecoratorGradientDescent.h"

class GradientDescentAdam : public DecoratorGradientDescent {
public:
    void subGrad(Matrix2D &weights, const Matrix2D &grad) override;

};



