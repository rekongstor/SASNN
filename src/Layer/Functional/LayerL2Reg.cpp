#include "../../../include/Layer/Functional/LayerL2Reg.h"

void LayerL2Reg::followProp() {
    multiplier.setCell(0, 0, right.getData()(0, 0) / static_cast<f32>(rowOriented ? left.getData().getRows() : left.getData().getCols()));
    data.MergeCellsOperator(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + r * r;
        }, nullptr, [](const f32 l) -> f32 {
            return l * l;
        });
    data.CellOperator(data, multiplier, [](const f32 l, const f32 r) -> f32 {
        return l * r;
        });
}

void LayerL2Reg::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D& g = *left.getGrad();
        g.CellOperator(left.getData(), multiplier, [](const f32 l, const f32 r) -> f32 {
            return 2.f * l * r;
            }, &grad);
    }
}

LayerL2Reg::LayerL2Reg(Layer& left, Layer& param, bool rowOriented) : LayerDynamic(1, 1),
left(left),
right(param),
multiplier(1, 1),
rowOriented(rowOriented) {}

