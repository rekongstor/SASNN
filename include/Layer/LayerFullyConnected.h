#pragma once


#include "Abstract/LayerDynamic.h"

class LayerFullyConnected : public LayerDynamic {
    Layer& left;
    Layer& right;
    void followProp() override;
    void backProp() override;
public:
    LayerFullyConnected(Layer &left, Layer &right);
};



