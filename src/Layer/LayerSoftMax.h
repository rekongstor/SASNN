#pragma once


#include "../../include/Layer/Abstract/LayerDynamic.h"
/**
 * Performs SoftMax function on each column (or row if transposed)
 */
class LayerSoftMax : public LayerDynamic {
    Layer &left;
    Matrix2D E;
    Matrix2D E_grad;
    Matrix2D Es;
    Matrix2D Es_grad;
    Matrix2D Es_gradF;
public:
    explicit LayerSoftMax(Layer &left, bool transpose = false);
    void followProp() override;
    void backProp() override;
};



