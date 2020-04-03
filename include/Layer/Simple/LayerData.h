#pragma once


#include "../Abstract/Layer.h"

class LayerData : public Layer {
    void followProp() override;
    void backProp() override;
    void clearGrad() override;
protected:
    void transposeData() override;
public:
    explicit LayerData(Matrix2D &data);
    void assignData(Matrix2D *d) override;
};



