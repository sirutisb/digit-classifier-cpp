#pragma once
#include "image.h"
#include <cstdint>
#include <initializer_list>
#include <vector>

enum class Activation {
    None,
    Sigmoid,
    // TODO: add more
};

struct LayerConfig {
    size_t size;
    Activation activation;
};

class Model {
public:
    Model(const std::initializer_list<LayerConfig>& config);
    void fit(const std::vector<LabeledImage>& train, int epochs, int batch_size, float learning_rate);
    uint8_t predict(const Image& im) const;
    void evaluate(const std::vector<LabeledImage>& test) const;
private:
    std::vector<std::vector<float>> weights_;
    std::vector<std::vector<float>> biases_;
    size_t layers_;
};