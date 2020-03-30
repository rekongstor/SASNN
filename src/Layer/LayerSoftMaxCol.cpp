#include "LayerSoftMaxCol.h"
#include <cmath>

void LayerSoftMaxCol::followProp() {
    E.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
        return expf(l);
    });
    Es.MergeRowsOperator(E, [](const f32 l, const f32 r) -> f32 {
        return r + l;
    });
    data.RowOperator(E, Es, [](const f32 l, const f32 r) -> f32 {
        return l / r;
    });
}

void LayerSoftMaxCol::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();

        Es_gradF.MergeRowsOperator(grad, [](const f32 l, const f32 r) -> f32 {
            return r + l;
        });
        Es_grad.RowOperator(E, Es, [](const f32 l, const f32 r) -> f32 {
            return -l / (r * r);
        }, &Es_gradF);

        E_grad.Clean();
        E_grad.RowOperator(E, Es, [](const f32 l, const f32 r) -> f32 {
            return 1.f / r;
        }, &grad);
        E_grad.RowOperator(E, Es_grad, [](const f32 l, const f32 r) -> f32 {
            return r;
        });
        g.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
            return expf(l);
        }, &E_grad);
    }
}

LayerSoftMaxCol::LayerSoftMaxCol(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                left(left),
                                                E(data.getRows(), data.getCols()),
                                                E_grad(data.getRows(), data.getCols(), true),
                                                Es(1, data.getCols()),
                                                Es_grad(1, data.getCols()),
                                                Es_gradF(1, data.getCols()) {}
