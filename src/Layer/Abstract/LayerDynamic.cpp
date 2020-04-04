#include <stdexcept>
#include "../../../include/Layer/Abstract/LayerDynamic.h"

void LayerDynamic::clearGrad() {
    grad.Clean();
}

LayerDynamic::LayerDynamic(size_t rows, size_t cols) : Layer(data),
                                                       data(rows, cols),
                                                       grad(rows, cols, true) {
}

Matrix2D *LayerDynamic::getGrad() {
    return &grad;
}

void LayerDynamic::transposeData() {
    data.Transpose();
}

void LayerDynamic::assignData(const Matrix2D *) {
    throw std::runtime_error("Unable to assign data to not a simple layer");
}
