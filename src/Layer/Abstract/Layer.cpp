#include "../../../include/Layer/Abstract/Layer.h"

 Matrix2D &Layer::getData()  {
    return *self;
}

Layer::Layer(Matrix2D &data) : self(&data) {

}

Matrix2D *Layer::getGrad() {
    return nullptr;
}

void Layer::subGrad(f32) {

}


