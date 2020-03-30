#include "LayerLeakyReLU.h"

void LayerLeakyReLU::followProp() {
    data.CellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
        return l <= 0.f ? l * r : l;
    });
}

void LayerLeakyReLU::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.CellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
            return l <= 0.f ? r : 1.f;
        }, &grad);
    }
    // right is a hyper-parameter and will not be changer with gradient descent
}

LayerLeakyReLU::LayerLeakyReLU(Layer &left, Layer &right) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                            left(left),
                                                            right(right) {

}
