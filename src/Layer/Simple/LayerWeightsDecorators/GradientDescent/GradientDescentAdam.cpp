#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdam.h"

void GradientDescentAdam::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    learningRate.setCell(0, 0, step);
}

GradientDescentAdam::GradientDescentAdam() : learningRate(1, 1) {}
