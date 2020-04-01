#pragma once


#include <memory>
#include "../DecoratorGradientDescent.h"

class GradientDescentRMSProp : public DecoratorGradientDescent {
    Matrix2D learningRate;
    std::shared_ptr<Matrix2D> accumulated;
    std::shared_ptr<Matrix2D> adaptiveLearningRate;
    Matrix2D rho;
public:
    explicit GradientDescentRMSProp(f32 rho = 0.9f);
    void subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) override;

};



