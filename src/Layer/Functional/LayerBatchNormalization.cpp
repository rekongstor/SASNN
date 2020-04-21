#include <cmath>
#include "../../../include/Layer/Functional/LayerBatchNormalization.h"

#define EPSILON 0.00001f

void LayerBatchNormalization::followProp() {
    auto Merge_functor = RowOriented ? &Matrix2D::MergeRowsOperator : &Matrix2D::MergeColsOperator;
    auto Data_functor = RowOriented ? &Matrix2D::RowOperator : &Matrix2D::ColOperator;

    // calculating mean
    (mean.*Merge_functor)(left.getData(), [](const f32 l, const f32 r) -> f32 {
        return l + r;
    }, nullptr, nullptr);
    mean.EachCellOperator(mean, size, [](const f32 l, const f32 r) -> f32 {
        return l / r;
    });

    // calculating variance
    (data.*Data_functor)(left.getData(), mean, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    }, nullptr);
    (variance.*Merge_functor)(data, [](const f32 l, const f32 r) -> f32 {
        return l + r * r;
    }, nullptr, [](const f32 l) -> f32 {
        return l * l;
    });
    variance.EachCellOperator(variance, size, [](const f32 l, const f32 r) -> f32 {
        return l / r;
        // variance + epsilon
    });

    // calculating x^
    xb.EachCellOperator(left.getData(), mean, variance, [](const f32 x, const f32 m, const f32 v) -> f32 {
        return (x - m) / (sqrtf(v + EPSILON));
    });

    // calculating data
    data.EachCellOperator(xb, gamma, beta, [](const f32 x, const f32 g, const f32 b) -> f32 {
        return x * g + b;
    });
}

void LayerBatchNormalization::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        auto Merge_functor = RowOriented ? &Matrix2D::MergeRowsOperator : &Matrix2D::MergeColsOperator;
        // dxb
        d_xb.EachCellOperator(grad, gamma, [](const f32 l, const f32 r) -> f32 {
            return l * r;
        });
        // d_gamma. reusing xb matrix
        xb.EachCellOperator(grad, xb, [](const f32 l, const f32 r) -> f32 {
            return l * r;
        });
        (d_gamma.*Merge_functor)(xb, [](const f32 l, const f32 r) -> f32 {
            return l + r;
        }, nullptr, nullptr);

        // d_beta
        (d_beta.*Merge_functor)(grad, [](const f32 l, const f32 r) -> f32 {
            return l + r;
        }, nullptr, nullptr);

        // d_variance. reusing xb matrix
        xb.EachCellOperator(grad, left.getData(), mean, [](const f32 g, const f32 x, const f32 m) -> f32 {
            return g * (x - m);
        });
        (d_variance.*Merge_functor)(xb, [](const f32 l, const f32 r) -> f32 {
            return l + r;
        }, nullptr, nullptr);
        d_variance.EachCellOperator(d_variance, gamma, variance, [](const f32 dv, const f32 g, const f32 v) -> f32 {
            return dv * (-g / 2.f / ((v + EPSILON) * sqrtf(v + EPSILON)));
        });

        // d_mean. reusing xb matrix
        xb.EachCellOperator(left.getData(), mean, [](const f32 l, const f32 r) -> f32 {
            return l - r;
        });
        (d_mean.*Merge_functor)(xb, [](const f32 l, const f32 r) -> f32 {
            return l + r;
        }, nullptr, nullptr);
        d_mean.EachCellOperator(d_mean, d_variance, size, [](const f32 dm, const f32 dv, const f32 s) -> f32 {
            return -2.f * dv / s * dm;
        });
        d_mean.EachCellOperator(d_mean, d_beta, gamma, variance, [](const f32 dm, const f32 db, const f32 g, const f32 v) -> f32 {
            return -db * g / sqrtf(v + EPSILON) + dm;
        });

        // grad. finally. also reusing xb
        xb.EachCellOperator(d_variance, left.getData(), mean, d_mean, [](const f32 dv, const f32 x, const f32 m, const f32 dm) -> f32 {
            return dv * 2.f * (x - m) + dm;
        });
        g.EachCellOperator(d_xb, variance, xb, size, [](const f32 dxb, const f32 v, const f32 xb, const f32 s) -> f32 {
            return dxb / sqrtf(v) + xb / s;
        });
    }
}

LayerBatchNormalization::LayerBatchNormalization(Layer &left, Layer& beta, Layer& gamma, bool rowOriented, f32 g, f32 b) :
        LayerDynamic(left.getData().getRows(), left.getData().getCols()),
        left(left),
        xb(left.getData().getRows(), left.getData().getCols()),
        d_xb(left.getData().getRows(), left.getData().getCols()),
        mean(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
        d_mean(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
        variance(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
        d_variance(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
        gamma(gamma.getData()),
        d_gamma(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
        beta(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
        d_beta(rowOriented ? 1 : left.getData().getRows(), rowOriented ? left.getData().getCols() : 1),
        size(1, 1),
        RowOriented(rowOriented) {
    gamma.Fill(g);
    beta.Fill(b);
    size.Fill(static_cast<f32>(RowOriented ? left.getData().getRows() : left.getData().getCols()));
}

void LayerBatchNormalization::subGrad(f32 step) {
    gradientDescent->subGrad(gamma, d_gamma, step);
    gradientDescent->subGrad(beta, d_beta, step);
}