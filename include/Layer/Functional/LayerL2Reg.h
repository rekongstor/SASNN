#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerL2Reg : public LayerDynamic {
    Layer &left;
    Layer &right;
    Matrix2D multiplier;
    bool rowOriented;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerL2Reg(Layer &left, Layer &param, bool rowOriented = true);
};



