#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"

class LayerSigmoid : public LayerDynamic {
    Layer &left;
public:
    explicit LayerSigmoid(Layer &left);
public:
    void followProp() override;
    void backProp() override;
};



