#include "../../include/Layer/LayerFullyConnected.h"

void LayerFullyConnected::followProp() {
    data.MultiplyOperator(left.getData(), right.getData());
}

void LayerFullyConnected::backProp() {

}

LayerFullyConnected::LayerFullyConnected(Layer &left, Layer &right) : LayerDynamic(left.getData().getRows(), right.getData().getCols()),
                                                                      left(left),
                                                                      right(right) {
}
