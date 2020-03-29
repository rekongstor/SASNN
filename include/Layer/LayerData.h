#pragma once


#include "Abstract/Layer.h"

class LayerData : public Layer {
    void followProp() override;
    void backProp() override;
    void clearGrad() override;
public:
    explicit LayerData(const Matrix2D& data);
};



