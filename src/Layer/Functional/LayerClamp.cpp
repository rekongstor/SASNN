#include "../../../include/Layer/Functional/LayerClamp.h"

void LayerClamp::followProp() {
    data.EachCellOperator(left.getData(), lower_bound, upper_bound, [](const f32 l, const f32 low, const f32 up) -> f32 {
        return std::max(low, std::min(up, l));
    });
}

void LayerClamp::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), lower_bound, upper_bound, [](const f32 l, const f32 low, const f32 up) -> f32 {
            if (l > low && l < up)
                return 1.f;
            return 0.f;
        }, &grad);
    }
}

LayerClamp::LayerClamp(Layer &left, f32 lowerBound, f32 upperBound) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                                      lower_bound(lowerBound),
                                                                      upper_bound(upperBound),
                                                                      left(left) {}
