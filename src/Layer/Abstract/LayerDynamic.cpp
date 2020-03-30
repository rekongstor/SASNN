#include "../../../include/Layer/Abstract/LayerDynamic.h"

void LayerDynamic::clearGrad() {
    grad.Clean();
}

LayerDynamic::LayerDynamic(size_t rows, size_t cols) : data(rows, cols),
                                                       grad(rows, cols, true),
                                                       Layer(data) {
}

Matrix2D *LayerDynamic::getGrad() {
    return &grad;
}

void LayerDynamic::transposeData() {
    data.transpose();
}
