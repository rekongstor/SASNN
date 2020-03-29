#pragma once


#include "Layer.h"

class LayerDynamic : public Layer {
protected:
    Matrix2D data;
    Matrix2D grad;
    LayerDynamic(size_t rows, size_t cols);
public:
    Matrix2D *getGrad() override;
private:
    void clearGrad() override;
};



