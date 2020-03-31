#pragma once


#include "../Abstract/LayerDynamic.h"
#include "../../Core/Matrix2D.h"

class LayerClassificationAccuracy : public LayerDynamic {
    Layer &left;
    Layer &right;
    Matrix2D maxProb;
    bool RowOriented;
public:
    LayerClassificationAccuracy(Layer &left, Layer &right, bool rowOriented);
    void followProp() override;
    void backProp() override;
};



