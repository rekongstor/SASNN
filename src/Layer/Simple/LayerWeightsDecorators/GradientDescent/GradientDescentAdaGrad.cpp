#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdaGrad.h"

void GradientDescentAdaGrad::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    learningRate.setCell(0, 0, step);
}

GradientDescentAdaGrad::GradientDescentAdaGrad() : learningRate(1, 1) {}
