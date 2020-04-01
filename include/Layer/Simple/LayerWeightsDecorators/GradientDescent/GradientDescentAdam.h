#pragma once


#include <memory>
#include "../DecoratorGradientDescent.h"

class GradientDescentAdam : public DecoratorGradientDescent {
    Matrix2D learningRate;
    std::shared_ptr<Matrix2D> accumulated;
    std::shared_ptr<Matrix2D> velocity;
    Matrix2D momentum;
    Matrix2D rho;
    void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) override;
public:
    explicit GradientDescentAdam(f32 momentum = 0.9, f32 rho = 0.9);
};



