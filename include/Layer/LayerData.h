#pragma once


#include "Layer.h"

class LayerData : public Layer {
    const Matrix2D& data;
    void followProp() override;
    void backProp(Matrix2D &grad) override;
    void clearGrad() override;
protected:
    const Matrix2D &getData() override;
public:
    explicit LayerData(const Matrix2D& data);
};



