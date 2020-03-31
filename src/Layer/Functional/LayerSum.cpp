#include "../../../include/Layer/Functional/LayerSum.h"

void LayerSum::followProp() {
    data.CellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
}

void LayerSum::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(data, [](const f32 l) -> f32 {
            return 1.f;
        }, &grad);
    }
    if (right.getGrad() != nullptr) {
        Matrix2D &g = *right.getGrad();
        g.EachCellOperator(data, [](const f32 l) -> f32 {
            return 1.f;
        }, &grad);
    }
}

LayerSum::LayerSum(Layer &left, Layer &right) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                left(left),
                                                right(right) {}
