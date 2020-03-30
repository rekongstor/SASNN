#pragma once


#include "../../include/Core/Random.h"
#include "../../include/Layer/Abstract/LayerDynamic.h"

class LayerWeights : public LayerDynamic {
    void backProp() override;
    void followProp() override;
public:
    LayerWeights(size_t rows, size_t cols, bool random = false);
    LayerWeights(size_t rows, size_t cols, f32 xavier_inputs);
};



