#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerSum : public LayerDynamic {
    Layer &left;
    Layer &right;
    void followProp() override;
    void backProp() override;
public:
    LayerSum(Layer &left, Layer &right);
};



