#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerClamp : public LayerDynamic {
    Layer &left;
    Matrix2D lower_bound;
    Matrix2D upper_bound;
public:
    LayerClamp(Layer &left, f32 lowerBound = 0.f, f32 upperBound = 1.f);
public:
    void followProp() override;
    void backProp() override;

};



