#pragma once


#include "../../../include/Layer/Abstract/LayerDynamic.h"

class LayerSumAll : public LayerDynamic {
    Layer &left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerSumAll(Layer &left);
};



