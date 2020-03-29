#include "../../include/stdafx.h"
#include <cmath>

f32 sum(f32 left, f32 right) {
    return left + right;
}

f32 sub(f32 left, f32 right) {
    return left - right;
}

f32 mul(f32 left, f32 right) {
    return left * right;
}

f32 div(f32 left, f32 right) {
    return left / right;
}

f32 inv(f32 left) {
    return 1.f / left;
}

f32 neg(f32 left) {
    return -left;
}

f32 ReLU(f32 left) {
    return left < 0.f ? 0.f : left;
}

f32 leakyReLU(f32 left, f32 right) {
    return left < 0.f ? left * right : left;
}

f32 sigmoid(f32 left) {
    return 1.f / (1.f + expf(left));
}
