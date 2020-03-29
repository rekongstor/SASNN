#include "LayerFullyConnected.h"

void LayerFullyConnected::followProp() {
    data.MultiplyOperator(left.getData(), right.getData());
}

void LayerFullyConnected::backProp() {
    // l[a x b] * r[b x c] = f[a x c]
    // Input data cannot be modified, so self gradient and input gradient will be transposed instead
    // Don't forget to return transposed matrices to normal
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad(); // [a x b]
        grad.transpose(); // [c x a]
        g.transpose(); // [b x a]
        g.MultiplyOperator(right.getData(), grad); // [b x c] * [c x a] = [b x a]
        grad.transpose();
        g.transpose();
    }
    if (right.getGrad() != nullptr) {
        Matrix2D &g = *right.getGrad(); // [b x c]
        grad.transpose(); // [c x a]
        g.transpose(); // [c x b]
        g.MultiplyOperator(grad, left.getData()); // [c x a] * [a x b] = [c x b]
        grad.transpose();
        g.transpose();
    }
}

LayerFullyConnected::LayerFullyConnected(Layer &left, Layer &right) : LayerDynamic(left.getData().getRows(), right.getData().getCols()),
                                                                      left(left),
                                                                      right(right) {
}
