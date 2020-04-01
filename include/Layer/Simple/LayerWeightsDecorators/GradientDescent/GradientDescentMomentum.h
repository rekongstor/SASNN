#pragma once


#include <memory>
#include "../DecoratorGradientDescent.h"

class GradientDescentMomentum : public DecoratorGradientDescent {
    std::shared_ptr<Matrix2D> velocity;
    Matrix2D momentum;
public:
    explicit GradientDescentMomentum(f32 velocity);
private:
    void subGrad(Matrix2D &weights, const Matrix2D &grad) override;
};



