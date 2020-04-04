#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerSoftMax : public LayerDynamic {
    Layer& left;
    Matrix2D Es;
    void followProp() override;
    void backProp() override;
    bool RowOriented;
public:
    explicit LayerSoftMax(Layer &left, bool rowOriented = true);
};



