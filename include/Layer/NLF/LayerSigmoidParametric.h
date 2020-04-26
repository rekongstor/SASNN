#pragma once

#include "../Abstract/LayerDynamic.h"

class LayerSigmoidParametric : public LayerDynamic {
    Layer &left;
    Layer &right;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerSigmoidParametric(Layer &left, Layer &right);
};



