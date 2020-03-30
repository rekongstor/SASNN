#include "LayerLeakyReLU.h"

void LayerLeakyReLU::followProp() {
    data.CellOperator(left.getData(), right.getData(), [](const f32 left, const f32 right) -> f32 {
        return left <= 0.f ? left * right : left;
    });
}

void LayerLeakyReLU::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.CellOperator(left.getData(), right.getData(), [](const f32 left, const f32 right) -> f32 {
            return left <= 0.f ? right : 1.f;
        }, &grad);
    }
    // right is a hyper-parameter and will not be changer with gradient descent
}

LayerLeakyReLU::LayerLeakyReLU(Layer &left, Layer &right) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                            left(left),
                                                            right(right) {

}
