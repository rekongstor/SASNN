#pragma once


#include <memory>
#include "../DecoratorGradientDescent.h"

class GradientDescentAdaGrad : public DecoratorGradientDescent {
    Matrix2D learningRate;
    std::shared_ptr<Matrix2D> accumulated;
    std::shared_ptr<Matrix2D> adaptiveLearningRate;
public:
    explicit GradientDescentAdaGrad();
    void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) override;

};



