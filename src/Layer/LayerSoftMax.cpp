#include "LayerSoftMax.h"
#include <cmath>

void LayerSoftMax::followProp() {
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

void LayerSoftMax::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();

        Es_gradF.MergeRowsOperator(grad, [](const f32 l, const f32 r) -> f32 {
            return r + l;
        });
        Es_grad.RowOperator(E, Es, [](const f32 l, const f32 r) -> f32 {
            f32 f1 = -l;
            f32 f2 = r*r;
            f32 f3 = f1 / f2;
            if (std::isnan(f3))
                return -l / powf(r,2.f);
            return -l / powf(r,2.f);
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

LayerSoftMax::LayerSoftMax(Layer &left, bool transpose) : LayerDynamic(!transpose ? left.getData().getRows() : left.getData().getCols(),
                                                                       !transpose ? left.getData().getCols() : left.getData().getRows()),
                                                          left(left),
                                                          E(!transpose ? data.getRows() : data.getCols(), !transpose ? data.getCols() : data.getRows()),
                                                          E_grad(!transpose ? data.getRows() : data.getCols(), !transpose ? data.getCols() : data.getRows(), true),
                                                          Es(!transpose ? 1 : data.getCols(), !transpose ? data.getCols() : 1),
                                                          Es_grad(!transpose ? 1 : data.getCols(), !transpose ? data.getCols() : 1),
                                                          Es_gradF(!transpose ? 1 : data.getCols(), !transpose ? data.getCols() : 1) {
    if (transpose)
        data.transpose();
}
