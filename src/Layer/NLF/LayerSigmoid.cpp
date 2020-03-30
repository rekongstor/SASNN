#include "../../../include/Layer/NLF/LayerSigmoid.h"
#include <cmath>

void LayerSigmoid::followProp() {
    data.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
        return 1.f / (1.f + expf(-l));
    });
}

void LayerSigmoid::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(data, [](const f32 l) -> f32 {
            return l * (1 - l);
        }, &grad);
    }
}

LayerSigmoid::LayerSigmoid(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                          left(left) {}
