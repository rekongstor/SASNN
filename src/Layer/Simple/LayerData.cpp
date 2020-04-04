#include "../../../include/Layer/Simple/LayerData.h"

LayerData::LayerData(const Matrix2D &data) : Layer(data) {

}


void LayerData::followProp() {

}

void LayerData::backProp() {

}

void LayerData::clearGrad() {

}

void LayerData::transposeData() {

}

void LayerData::assignData(const Matrix2D *d) {
    self = d;
}
