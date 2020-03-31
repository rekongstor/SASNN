#include "LayerGT.h"

void LayerGT::followProp() {
    data.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
        return l * r;
    });
}

void LayerGT::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
            return r;
        }, &grad);
    }
    if (right.getGrad() != nullptr) {
        Matrix2D &g = *right.getGrad();
        g.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
            return l;
        }, &grad);
    }
}

LayerGT::LayerGT(Layer &left, Layer &right) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                              left(left),
                                              right(right) {}
