#pragma once


#include "Layer.h"

class LayerDynamic : public Layer {
protected:
    Matrix2D data;
    Matrix2D grad;
    LayerDynamic(size_t rows, size_t cols);
    void transposeData() final;
public:
    Matrix2D *getGrad() final;
    void assignData(Matrix2D *d) override;
private:
    void clearGrad() override ;
};



