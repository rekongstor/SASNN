#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerReLU : public LayerDynamic {
    Layer &left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerReLU(Layer &left);
};



