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

struct TrainHistory {
    int epoch;
    float epoch_loss;
    float epoch_accuracy;
};

class Model {
public:
    Model(const std::initializer_list<LayerConfig>& config);
    std::vector<TrainHistory> fit(const std::vector<LabeledImage>& train, int epochs, int batch_size, float learning_rate);
    uint8_t predict(const Image& im) const;
    void evaluate(const std::vector<LabeledImage>& test) const;
private:
    std::vector<float> forwardPass(const Image& image) const;

    std::vector<std::vector<float>> weights_;
    std::vector<std::vector<float>> biases_;
    std::vector<unsigned int> layerSizes_;

    friend int main();
};