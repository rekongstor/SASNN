#pragma once


#include "../../include/Layer/Abstract/Layer.h"

class LayerData : public Layer {
    void followProp() override;
    void backProp() override;
    void clearGrad() override;
protected:
    void transposeData() override;
public:
    explicit LayerData(const Matrix2D& data);
};



