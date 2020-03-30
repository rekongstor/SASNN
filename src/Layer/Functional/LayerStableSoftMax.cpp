#include "LayerStableSoftMax.h"
#include <cmath>

void LayerStableSoftMax::followProp() {
    auto Es_functor = ColumnOriented ? &Matrix2D::MergeColsOperator : &Matrix2D::MergeRowsOperator;
    (Es.*Es_functor)(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + expf(r);
    }, nullptr, [](const f32 l) -> f32 {
        return expf(l);
    });
    auto data_functor = ColumnOriented ? &Matrix2D::ColOperator : &Matrix2D::RowOperator;
    (data.*data_functor)(left.getData(), Es, [](const f32 l, const f32 r) -> f32 {
        return exp(l) / r;
    }, nullptr);
}

void LayerStableSoftMax::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(data, [](const f32 l) -> f32 {
            return l * (1.f - l);
        }, &grad);
    }
}

LayerStableSoftMax::LayerStableSoftMax(Layer &left, bool RowOriented) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                                        left(left),
                                                                        Es(RowOriented ? left.getData().getRows() : 1,
                                                                           RowOriented ? 1 : left.getData().getRows()),
                                                                        ColumnOriented(RowOriented) {

}