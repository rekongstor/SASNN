#include "../../../include/Layer/NLF/LayerSigmoidParametric.h"
#include <cmath>

void LayerSigmoidParametric::followProp() {
    data.EachCellOperator(left.getData(), right.getData(), [](const f32 l, const f32 r) -> f32 {
        return 1.f / (1.f + expf(-r * l));
    });
}

void LayerSigmoidParametric::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(data, right.getData(), [](const f32 l, const f32 r) -> f32 {
            return l * (1 - l) * r;
        }, &grad);
    }
    if (right.getGrad() != nullptr) {
        Matrix2D &g = *right.getGrad();
        g.EachCellOperator(data, left.getData(), [](const f32 l, const f32 r) -> f32 {
            return l * (1 - l) * r;
        }, &grad);
    }
}

LayerSigmoidParametric::LayerSigmoidParametric(Layer &left, Layer &right) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                                            left(left),
                                                                            right(right) {}
