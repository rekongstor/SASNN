#include "../../../include/Layer/NLF/LayerReLU.h"

void LayerReLU::followProp() {
    data.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
        return l <= 0.f ? 0.f : l;
    });
}

void LayerReLU::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
            return l <= 0.f ? 0.f : 1.f;
        }, &grad);
    }
}

LayerReLU::LayerReLU(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                    left(left) {

}
