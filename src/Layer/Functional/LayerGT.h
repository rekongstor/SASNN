#pragma once


#include "../../../include/Layer/Abstract/LayerDynamic.h"

class LayerGT : public LayerDynamic {
    Layer& left;
    Layer& right;
    void followProp() override;
    void backProp() override;
public:
    LayerGT(Layer &left, Layer &right);
};



