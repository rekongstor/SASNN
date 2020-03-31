#include "../../../include/Layer/Functional/LayerStableSoftMax.h"
#include <cmath>

void LayerStableSoftMax::followProp() {
    auto Merge_functor = ColumnOriented ? &Matrix2D::MergeColsOperator : &Matrix2D::MergeRowsOperator;
    auto Data_functor = ColumnOriented ? &Matrix2D::ColOperator : &Matrix2D::RowOperator;
    (maxData.*Merge_functor)(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return fmaxf(l, r);
    }, nullptr, nullptr);
    (normalizedData.*Data_functor)(left.getData(), maxData, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    }, nullptr);

    (Es.*Merge_functor)(normalizedData, [](const f32 l, const f32 r) -> f32 {
        return l + expf(r);
    }, nullptr, [](const f32 l) -> f32 {
        return expf(l);
    });
    (data.*Data_functor)(normalizedData, Es, [](const f32 l, const f32 r) -> f32 {
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
                                                                           RowOriented ? 1 : left.getData().getCols()),
                                                                        ColumnOriented(RowOriented),
                                                                        maxData(RowOriented ? left.getData().getRows() : 1,
                                                                                RowOriented ? 1 : left.getData().getCols()),
                                                                        normalizedData(RowOriented ? left.getData().getRows() : left.getData().getCols(),
                                                                                       RowOriented ? left.getData().getCols() : left.getData().getRows()) {

}