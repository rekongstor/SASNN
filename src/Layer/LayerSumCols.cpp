#include "LayerSumCols.h"

void LayerSumCols::followProp() {
    data.MergeColsOperator(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
}

void LayerSumCols::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.ColOperator(left.getData(), grad, [](const f32 l, const f32 r) -> f32 {
            return r;
        });
    }
}

LayerSumCols::LayerSumCols(Layer &left) : LayerDynamic(left.getData().getRows(), 1),
                                          left(left) {}
