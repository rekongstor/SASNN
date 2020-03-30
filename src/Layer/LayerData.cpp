#include "LayerData.h"
#include <stdexcept>

LayerData::LayerData(const Matrix2D &data) : Layer(data) {

}

void LayerData::followProp() {

}

void LayerData::backProp() {

}

void LayerData::clearGrad() {

}

void LayerData::transposeData() {
#if (DEBUG_LEVEL > 0)
    throw std::runtime_error("Constant data could not be transposed");
#endif
}
