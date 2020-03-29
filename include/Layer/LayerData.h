#pragma once


#include "Layer.h"

class LayerData : Layer {
    const Matrix2D& data;
    explicit LayerData(const Matrix2D& data);
    void followProp() override;
    void backProp(Matrix2D &grad) override;
    void clearGrad() override;
};



