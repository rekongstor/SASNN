#pragma once


#include "Layer.h"
#include "../../src/Core/Random.h"

class LayerWeights : Layer{
    Matrix2D data;
    Matrix2D grad;
    void backProp(Matrix2D &grad) override;
    void clearGrad() override;
    void followProp() override;
    LayerWeights(size_t rows, size_t cols, bool random = false);
    LayerWeights(size_t rows, size_t cols, size_t xavier_inputs);
};



