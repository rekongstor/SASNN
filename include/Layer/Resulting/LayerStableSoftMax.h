#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerStableSoftMax : public LayerDynamic {
    Layer &left;
    Matrix2D Es;
    Matrix2D maxData;
    Matrix2D normalizedData;
    bool RowOriented;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerStableSoftMax(Layer &left, bool rowOriented = true);
};



