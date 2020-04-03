#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerBatchNormalization : public LayerDynamic {
Layer& left;
Matrix2D mean;
Matrix2D dev;
Matrix2D gamma;
Matrix2D beta;
Matrix2D size;
bool RowOriented;
public:
    explicit LayerBatchNormalization(Layer &left, bool rowOriented = true, f32 g = 1.f, f32 b = 0.f);
private:
    void followProp() override;
    void backProp() override;
};


