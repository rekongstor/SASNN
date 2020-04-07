#pragma once


#include <memory>
#include "../DecoratorGradientDescent.h"

class GradientDescentAdaGrad : public DecoratorGradientDescent {
    Matrix2D learningRate;
    std::shared_ptr<Matrix2D> accumulated;
public:
    GradientDescentAdaGrad();
    void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) override;

};



