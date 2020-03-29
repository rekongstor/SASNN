#pragma once


#include "Layer.h"
#include "../Core/Random.h"

class LayerWeights : public Layer{
protected:
    const Matrix2D &getData() override;
private:
    Matrix2D data;
    Matrix2D grad;
    void backProp(Matrix2D &grad) override;
    void clearGrad() override;
    void followProp() override;
public:
    LayerWeights(size_t rows, size_t cols, bool random = false);
    LayerWeights(size_t rows, size_t cols, f32 xavier_inputs);
};



