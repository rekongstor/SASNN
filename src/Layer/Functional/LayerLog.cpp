#include "LayerLog.h"
#include <cmath>

void LayerLog::followProp() {
    data.EachCellOperator(left.getData(),[](const f32 l) -> f32 {
        return logf(l);
    });
}

void LayerLog::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(left.getData(), [](const f32 l) -> f32 {
            return 1.f / l;
        }, &grad);
    }
}

LayerLog::LayerLog(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                  left(left) {}
