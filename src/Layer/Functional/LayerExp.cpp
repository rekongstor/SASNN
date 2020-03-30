#include "../../../include/Layer/Functional/LayerExp.h"
#include <cmath>

void LayerExp::followProp() {
    data.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
        return expf(l);
    });
}

void LayerExp::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
            return expf(l);
        }, &grad);
    }
}

LayerExp::LayerExp(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                  left(left) {}
