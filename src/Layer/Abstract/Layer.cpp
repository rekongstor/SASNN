#include "../../../include/Layer/Abstract/Layer.h"

const Matrix2D &Layer::getData() const {
    return self;
}

Layer::Layer(const Matrix2D &data) : self(data) {

}

Matrix2D *Layer::getGrad() {
    return nullptr;
}

void Layer::subGrad() {

}


