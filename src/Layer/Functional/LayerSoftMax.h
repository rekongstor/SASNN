#pragma once


#include "../../../include/Layer/Abstract/LayerDynamic.h"

class LayerSoftMax : public LayerDynamic {
    Layer& left;
    Matrix2D Es;
    void followProp() override;
    void backProp() override;
    bool RowOriented;
public:
    LayerSoftMax(Layer &left, bool rowOriented);
};



