#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerLeastSquaresRegression : public LayerDynamic {

    Layer &left;
    Layer &right;
    Matrix2D SquaredDiff;
public:
    LayerLeastSquaresRegression(Layer &left, Layer &GT);
public:
    void followProp() override;
    void backProp() override;
};



