#include <limits>
#include "../../../include/Layer/Accuracy/LayerClassificationAccuracy.h"

void LayerClassificationAccuracy::followProp() {
    auto Data_functor = RowOriented ? &Matrix2D::FindRowOperator : &Matrix2D::FindColOperator;
    (maxProb.*Data_functor)(left.getData(), std::numeric_limits<f32>::min(), [](const f32 l, const f32 r) -> bool {
        return (l > r);
    }, &right.getData());
    data.MergeCellsOperator(maxProb,[](const f32 l, const f32 r) -> f32{
        return l + r;
    });
}

void LayerClassificationAccuracy::backProp() {

}

LayerClassificationAccuracy::LayerClassificationAccuracy(Layer &left, Layer &right, bool rowOriented) : LayerDynamic(1, 1),
                                                                                                        left(left),
                                                                                                        right(right),
                                                                                                        maxProb(left.getData().getRows(), left.getData().getCols()),
                                                                                                        RowOriented(rowOriented) {}
