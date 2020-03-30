#include "LayerTanh.h"
#include <cmath>

void LayerTanh::followProp() {
    data.EachCellOperator(left.getData(), [](const f32 left) -> f32 {
        return tanhf(left) * 0.5f + 0.5f;
    });
}

void LayerTanh::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), [](const f32 left) -> f32 {
            return (1.f - tanhf(left) * tanhf(left)) * 0.5f;
        }, &grad);
    }
}

LayerTanh::LayerTanh(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                    left(left) {}
