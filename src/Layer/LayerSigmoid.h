#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"

class LayerSigmoid : public LayerDynamic {
    Layer &left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerSigmoid(Layer &left);
};



