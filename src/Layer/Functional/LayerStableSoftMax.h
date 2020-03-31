#pragma once


#include "../../../include/Layer/Abstract/LayerDynamic.h"

class LayerStableSoftMax : public LayerDynamic {
    Layer &left;
    Matrix2D Es;
    Matrix2D maxData;
    Matrix2D normalizedData;
    bool ColumnOriented;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerStableSoftMax(Layer &left, bool RowOriented = true);
};



