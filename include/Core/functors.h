#pragma once

#include "../stdafx.h"

extern f32 sum(f32 left, f32 right);
extern f32 sub(f32 left, f32 right);
extern f32 mul(f32 left, f32 right);
extern f32 div(f32 left, f32 right);
extern f32 inv(f32 left);
extern f32 neg(f32 left);
extern f32 ReLU(f32 left);
extern f32 leakyReLU(f32 left, f32 right);
extern f32 sigmoid(f32 left);
