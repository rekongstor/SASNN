#include "LayerSoftMaxRow.h"
#include <cmath>

void LayerSoftMaxRow::followProp() {
    E.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
        return expf(l);
    });
    Es.MergeColsOperator(E, [](const f32 l, const f32 r) -> f32 {
        return r + l;
    });
    data.RowOperator(E, Es, [](const f32 l, const f32 r) -> f32 {
        return l / r;
    });
}

void LayerSoftMaxRow::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        E_grad.Clean();
        Es_grad.Clean();

        Es_grad.RowOperator(E, Es, [](const f32 l, const f32 r) -> f32 {
            return -l / (r * r);
        }, &grad);
        E_grad.RowOperator(E, Es, [](const f32 l, const f32 r) -> f32 {
            return 1.f / r;
        }, &grad);
        E_grad.EachCellOperator(E,[](const f32 l) -> f32{
            return 1.f;
        }, &Es_grad);
        g.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
            return expf(l);
        }, &E_grad);

    }
}

LayerSoftMaxRow::LayerSoftMaxRow(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                left(left),
                                                E(data.getRows(), data.getCols()),
                                                E_grad(data.getRows(), data.getCols()),
                                                Es(data.getRows(), 1),
                                                Es_grad(data.getRows(), 1) {}
