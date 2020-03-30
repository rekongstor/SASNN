#pragma once


#include "../../Core/Random.h"
#include "../Abstract/LayerDynamic.h"

class LayerWeights : public LayerDynamic {
public:
    void subGrad() override;
private:
    void backProp() override;
    void followProp() override;
public:
    LayerWeights(size_t rows, size_t cols, bool random = false);
    LayerWeights(size_t rows, size_t cols, f32 xavier_inputs);
};



