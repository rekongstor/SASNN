
#include "../../include/Core/functors.h"

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
