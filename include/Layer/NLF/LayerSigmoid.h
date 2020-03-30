#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerSigmoid : public LayerDynamic {
    Layer &left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerSigmoid(Layer &left);
};



