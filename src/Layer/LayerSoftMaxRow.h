#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"
/**
 * Performs SoftMax function on each row
 */
class LayerSoftMaxRow : public LayerDynamic {
    Layer &left;
    Matrix2D E;
    Matrix2D E_grad;
    Matrix2D Es;
    Matrix2D Es_grad;
public:
    explicit LayerSoftMaxRow(Layer &left);
    void followProp() override;
    void backProp() override;
};



