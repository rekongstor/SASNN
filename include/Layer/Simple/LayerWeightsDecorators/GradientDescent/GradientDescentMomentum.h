#pragma once


#include <memory>
#include "../DecoratorGradientDescent.h"

class GradientDescentMomentum : public DecoratorGradientDescent {
    Matrix2D learningRate;
    std::shared_ptr<Matrix2D> velocity;
    Matrix2D momentum;
    void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) override;
public:
    explicit GradientDescentMomentum(f32 velocity);
};



