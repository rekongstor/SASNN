#include "../../../include/Layer/Functional/LayerStableSoftMax.h"
#include <cmath>

void LayerStableSoftMax::followProp() {
    auto Merge_functor = RowOriented ? &Matrix2D::MergeColsOperator : &Matrix2D::MergeRowsOperator;
    auto Data_functor = RowOriented ? &Matrix2D::ColOperator : &Matrix2D::RowOperator;
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
        return expf(l) / r;
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

LayerStableSoftMax::LayerStableSoftMax(Layer &left, bool rowOriented) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                                        left(left),
                                                                        Es(rowOriented ? left.getData().getRows() : 1,
                                                                           rowOriented ? 1 : left.getData().getCols()),
                                                                        maxData(rowOriented ? left.getData().getRows() : 1,
                                                                                rowOriented ? 1 : left.getData().getCols()),
                                                                        normalizedData(rowOriented ? left.getData().getRows() : left.getData().getCols(),
                                                                                       rowOriented ? left.getData().getCols() : left.getData().getRows()),
                                                                        RowOriented(rowOriented){

}