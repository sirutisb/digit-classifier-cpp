#pragma once
#include "image.h"
#include "layer.h"

class Model {
public:
    Model(const std::initializer_list<LayerConfig>& config, float lr = 0.001);
    void fit(const std::vector<LabeledImage>& data);
    uint8_t predict(const Image& im) const;
private:

    static std::function<float(float)> get_activation(Activation activation);
    std::vector<Layer> layers_;
    float lr_;
};
