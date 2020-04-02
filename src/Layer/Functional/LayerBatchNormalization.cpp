#include <cmath>
#include "../../../include/Layer/Functional/LayerBatchNormalization.h"

void LayerBatchNormalization::followProp() {
    auto Merge_functor = RowOriented ? &Matrix2D::MergeRowsOperator : &Matrix2D::MergeColsOperator;
    auto Data_functor = RowOriented ? &Matrix2D::RowOperator : &Matrix2D::ColOperator;

    (mean.*Merge_functor)(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + r;
    }, nullptr, nullptr);
    mean.EachCellOperator(mean, size, [](const f32 l, const f32 r) -> f32 {
        return l / r;
    });
    (data.*Data_functor)(left.getData(), mean, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    }, nullptr);
    (dev.*Merge_functor)(data, [](const f32 l, const f32 r) -> f32 {
        return l + r * r;
    }, nullptr, [](const f32 l) -> f32 {
        return l * l;
    });
    dev.EachCellOperator(dev, size, [](const f32 l, const f32 r) -> f32 {
        return l / r + 0.00001f;
        // dev + epsilon
    });
    beta.EachCellOperator(beta, gamma, [](const f32 l, const f32 r) -> f32 {
        return l / r;
    });
    data.EachCellOperator(data, mean, dev, beta, [](const f32 x, const f32 m, const f32 d, const f32 b) -> f32 {
        return (x) / sqrtf(d) + b;
    }, &gamma);
    beta.EachCellOperator(beta, gamma, [](const f32 l, const f32 r) -> f32 {
        return l / r;
    });
}

void LayerBatchNormalization::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(gamma, [](const f32 l) -> f32 {
            return l;
        }, &grad);
        // reusing matrices
        auto Merge_functor = RowOriented ? &Matrix2D::MergeRowsOperator : &Matrix2D::MergeColsOperator;
        data.EachCellOperator(data, grad, gamma, beta, [](const f32 d, const f32 g, const f32 y, const f32 b) -> f32 {
            return g * (d - b) / y;
        });
        (dev.*Merge_functor)(data, [](const f32 l, const f32 r) -> f32 {
            return l + r;
        }, nullptr, nullptr); // gamma gradient
        gamma.EachCellOperator(gamma, dev, [](const f32 l, const f32 r) -> f32 {
            return l - r;
        });
        (dev.*Merge_functor)(data, [](const f32 l, const f32 r) -> f32 {
            return l + r;
        }, nullptr, nullptr); // beta gradient
        gamma.EachCellOperator(gamma, dev, [] (const f32 l, const f32 r) -> f32 {
            return l - r;
        });
    }
}

LayerBatchNormalization::LayerBatchNormalization(Layer &left, bool rowOriented, f32 g, f32 b) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                                                                                left(left),
                                                                                                mean(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
                                                                                                dev(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
                                                                                                gamma(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
                                                                                                beta(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
                                                                                                size(1, 1),
                                                                                                RowOriented(rowOriented) {
    gamma.Fill(g);
    beta.Fill(b);
    size.Fill(static_cast<f32>(RowOriented ? left.getData().getRows() : left.getData().getCols()));
}
