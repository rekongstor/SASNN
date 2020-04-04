#include "../../../include/Layer/NLF/LayerLeakyReLU.h"

void LayerLeakyReLU::followProp() {
    data.CellOperator(left.getData(), leak, [](const f32 l, const f32 r) -> f32 {
        return l <= 0.f ? l * r : l;
    });
}

void LayerLeakyReLU::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.CellOperator(left.getData(), leak, [](const f32 l, const f32 r) -> f32 {
            return l <= 0.f ? r : 1.f;
        }, &grad);
    }
}

LayerLeakyReLU::LayerLeakyReLU(Layer &left, f32 leak) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                            leak(leak),
                                                            left(left) {

}
