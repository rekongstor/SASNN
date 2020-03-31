#pragma once


#include "../../Core/Random.h"
#include "../Abstract/LayerDynamic.h"

class LayerWeights : public LayerDynamic {
    void backProp() override;
    void followProp() override;
    void assignData(const Matrix2D *d) override;

    void subGrad() override;
    Matrix2D gradLength;
    void normalizeGrad();
public:
    LayerWeights(size_t rows, size_t cols, bool random = false);
    LayerWeights(size_t rows, size_t cols, f32 xavier_inputs);
};



