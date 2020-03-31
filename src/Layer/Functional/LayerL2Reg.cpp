#include "../../../include/Layer/Functional/LayerL2Reg.h"

void LayerL2Reg::followProp() {
    data.MergeCellsOperator(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + r * r;
    }, nullptr, [](const f32 l) -> f32 {
        return l * l;
    });
}

void LayerL2Reg::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
            return 2.f * l;
        }, &grad);
    }
}

LayerL2Reg::LayerL2Reg(Layer &left) : LayerDynamic(1, 1),
                                      left(left) {}

