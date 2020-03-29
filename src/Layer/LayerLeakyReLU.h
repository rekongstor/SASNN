#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"

class LayerLeakyReLU : public LayerDynamic {
    Layer& left;
    Layer& right;
public:
    LayerLeakyReLU(Layer &left, Layer &right);
public:
    void followProp() override;
    void backProp() override;
};



