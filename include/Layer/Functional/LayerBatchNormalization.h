#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerBatchNormalization : public LayerDynamic {
    Layer &left;
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
public:
    explicit LayerBatchNormalization(Layer &left, bool rowOriented = true, f32 g = 1.f, f32 b = 0.f);
    void subGrad(f32 step) override;
private:
    void followProp() override;
    void backProp() override;
};



