#pragma once


#include "../../../include/Layer/Abstract/LayerDynamic.h"

class LayerCrossEntropyLoss : public LayerDynamic {
    Layer &left;
    Layer &right;
    Matrix2D NegativeLogGT;
    void followProp() override;
    void backProp() override;
public:
    LayerCrossEntropyLoss(Layer &left, Layer &GT);
};



