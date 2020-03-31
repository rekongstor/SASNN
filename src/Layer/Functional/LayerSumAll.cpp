#include "LayerSumAll.h"

void LayerSumAll::followProp() {
    data.MergeCellsOperator(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
}

void LayerSumAll::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.CellOperator(left.getData(), grad, [](const f32 l, const f32 r) -> f32 {
            return r;
        });
    }
}

LayerSumAll::LayerSumAll(Layer &left) : LayerDynamic(1, 1),
                                        left(left) {}
