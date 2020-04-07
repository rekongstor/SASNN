#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerRegressionAccuracy : public LayerDynamic {
    Layer &left;
    Layer &right;
    Matrix2D diff;
    bool RowOriented;
    void followProp() override;
    void backProp() override;
public:
    LayerRegressionAccuracy(Layer &left, Layer &right, bool rowOriented);
};



