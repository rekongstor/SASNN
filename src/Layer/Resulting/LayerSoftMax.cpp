#include <cmath>
#include "../../../include/Layer/Resulting/LayerSoftMax.h"

void LayerSoftMax::followProp() {
    auto Merge_functor = RowOriented ? &Matrix2D::MergeColsOperator : &Matrix2D::MergeRowsOperator;
    auto Data_functor = RowOriented ? &Matrix2D::ColOperator : &Matrix2D::RowOperator;
    (Es.*Merge_functor)(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + expf(r);
    }, nullptr, [](const f32 l) -> f32 {
        return expf(l);
    });
    (data.*Data_functor)(left.getData(), Es, [](const f32 l, const f32 r) -> f32 {
        return expf(l) / r;
    }, nullptr);
}

void LayerSoftMax::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(data, [](const f32 l) -> f32 {
            return l * (1.f - l);
        }, &grad);
    }
}

LayerSoftMax::LayerSoftMax(Layer &left, bool rowOriented) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                            left(left),
                                                            Es(rowOriented ? left.getData().getRows() : 1,
                                                                                         rowOriented ? 1 : left.getData().getCols()),
                                                            RowOriented(rowOriented) {}
