#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerSoftMax : public LayerDynamic {
    Layer& left;
    Matrix2D Es;
    bool RowOriented;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerSoftMax(Layer &left, bool rowOriented = true);
};



