#pragma once


#include "../DecoratorGradientDescent.h"

class GradientDescentStochastic : public DecoratorGradientDescent {
    Matrix2D learningRate;
    void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) override;
public:
    GradientDescentStochastic();
};



