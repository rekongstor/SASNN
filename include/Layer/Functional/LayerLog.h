#pragma once


#include "../Abstract/LayerDynamic.h"

class LayerLog : public LayerDynamic {
    Layer& left;
    void followProp() override;
    void backProp() override;
public:
    explicit LayerLog(Layer &left);
};



