#include "../../../include/Layer/Accuracy/LayerRegressionAccuracy.h"
#include <cmath>

void LayerRegressionAccuracy::followProp() {
    diff.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
        return (l - r) * (l - r);
    });
    data.MergeCellsOperator(diff, [](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
    data.setCell(0, 0, (data(0, 0) / static_cast<f32>(RowOriented ? left.getData().getCols() : left.getData().getRows())));
}

void LayerRegressionAccuracy::backProp() {

}

LayerRegressionAccuracy::LayerRegressionAccuracy(Layer &left, Layer &right, bool rowOriented) : LayerDynamic(1, 1),
                                                                                                left(left),
                                                                                                right(right),
                                                                                                diff(left.getData().getRows(), left.getData().getCols()),
                                                                                                RowOriented(rowOriented) {}
