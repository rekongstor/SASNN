#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerL2Reg : public LayerDynamic {
    Layer &left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerL2Reg(Layer &left);
};



