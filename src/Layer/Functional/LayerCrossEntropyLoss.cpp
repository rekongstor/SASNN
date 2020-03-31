#include "../../../include/Layer/Functional/LayerCrossEntropyLoss.h"
#include <cmath>

void LayerCrossEntropyLoss::followProp() {
    NegativeLogGT.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
        return -logf(l) * r;
    });
    data.MergeCellsOperator(NegativeLogGT,[](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
}

void LayerCrossEntropyLoss::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
            return -1.f / l * r;
        },&grad);
    }
}

LayerCrossEntropyLoss::LayerCrossEntropyLoss(Layer &left, Layer &GT) : LayerDynamic(1, 1),
                                                                       left(left),
                                                                       right(GT),
                                                                       NegativeLogGT(left.getData().getRows(), left.getData().getCols()) {}
