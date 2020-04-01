#pragma once


#include "../DecoratorGradientDescent.h"

class GradientDescentAdaGrad : public DecoratorGradientDescent {
    Matrix2D learningRate;
public:
    GradientDescentAdaGrad();
    void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) override;

};



