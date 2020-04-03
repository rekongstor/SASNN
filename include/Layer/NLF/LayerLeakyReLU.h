#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerLeakyReLU : public LayerDynamic {
    Matrix2D leak;
    Layer& left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerLeakyReLU(Layer& left, f32 leak = 0.01f);
};



