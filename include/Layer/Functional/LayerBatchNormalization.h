#pragma once


#include <memory>
#include "../Abstract/LayerDynamic.h"
#include "../Simple/LayerWeightsDecorators/DecoratorGradientDescent.h"

class LayerBatchNormalization : public LayerDynamic {
    Layer& left;
    Matrix2D xb;
    Matrix2D d_xb;
    Matrix2D mean;
    Matrix2D d_mean;
    Matrix2D variance;
    Matrix2D d_variance;
    Matrix2D gamma;
    Matrix2D d_gamma;
    Matrix2D beta;
    Matrix2D d_beta;
    Matrix2D size;
    bool RowOriented;
    std::shared_ptr<DecoratorGradientDescent> gradientDescent;
public:
    explicit LayerBatchNormalization(Layer& left, DecoratorGradientDescent *decoratorGradientDescent, bool rowOriented = true, f32 g = 1.f, f32 b = 0.f);
    void subGrad(f32 step) override;
private:
    void followProp() override;
    void backProp() override;
};



