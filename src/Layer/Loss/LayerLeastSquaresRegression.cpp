#include "../../../include/Layer/Loss/LayerLeastSquaresRegression.h"

void LayerLeastSquaresRegression::followProp() {
    SquaredDiff.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
        return (l - r) * (l - r);
    });
    data.MergeCellsOperator(SquaredDiff, [](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
}

void LayerLeastSquaresRegression::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
            return 2.f * (l - r);
        }, &grad);
    }
    if (right.getGrad() != nullptr) {
        Matrix2D &g = *right.getGrad();
        g.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
            return -2.f * (l - r);
        }, &grad);
    }
}

LayerLeastSquaresRegression::LayerLeastSquaresRegression(Layer &left, Layer &GT) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                                                   left(left),
                                                                                   right(GT),
                                                                                   SquaredDiff(left.getData().getRows(), left.getData().getCols()) {

}
