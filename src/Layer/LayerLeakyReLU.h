#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"

class LayerLeakyReLU : public LayerDynamic {
    Layer& left;
    Layer& right;
    void followProp() override;
    void backProp() override;
public:
    LayerLeakyReLU(Layer &left, Layer &right);
};



