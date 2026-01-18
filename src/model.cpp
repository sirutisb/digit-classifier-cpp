#include "model.h"
#include <cmath>
#include <cassert>
#include <print>

Model::Model(const std::initializer_list<LayerConfig>& config, float lr) : lr_(lr)
{
    layers_.reserve(config.size() - 1);
    for (auto it = config.begin();;) {
        auto next = std::next(it);
        if (next == config.end()) break;

        auto fn = get_activation(next->activation);
        layers_.emplace_back(it->size, next->size, fn);
        it = next;
    }
}

void Model::fit(const std::vector<LabeledImage>& data) {
    // training loop
}

uint8_t Model::predict(const Image& image) const {
    // Forward Pass

    // Convert image to initial activations
    std::vector<float> activations(&image[0][0], &image[0][0] + 784);


    for (const Layer& layer : layers_) {
        std::println("\nLayer {},{}", layer.inputSize, layer.outputSize);

        std::println("\nActivations:");
        for (const auto& a : activations) {
            std::println("{}, ", a);
        }
        std::println("\n\nWeights:");
        for (const auto& w : layer.weights) {
            std::print("{}, " , w);
        }
        std::println("\n\nNext Activations:");

        std::vector<float> nextActivations(layer.outputSize, 0.0f);
        for (size_t i = 0; i < layer.outputSize; ++i) {
            for (size_t j = 0; j < layer.inputSize; ++j) {
                nextActivations[i] += activations[j] * layer.weights[i * layer.inputSize + j];
            }
            nextActivations[i] += layer.biases[i];
            nextActivations[i] = layer.fn(nextActivations[i]);
            std::println("{}", nextActivations[i]);
        }
        activations = std::move(nextActivations);
    }

    float max_pred = 0;
    uint8_t pred_digit = 0;

    std::println("\nPredictions:");
    for (size_t i = 0; i < activations.size(); ++i) {
        float y = activations[i];
        std::println("{}: {} ", i, y);
        if (y > max_pred) {
            max_pred = y;
            pred_digit = (uint8_t)i;
        }
    }

    std::println("\nPredicting: {}", pred_digit);
    return pred_digit;
}

static std::function<float(float)> get_activation(Activation activation) {
    auto none_ = [](float x) { return x; };
    auto relu_ = [](float x) { return std::max(x, 0.f); };
    auto sigmoid_ = [](float x) { return 1.0f / (1.0f + expf(-x)); };
    auto tanh_ = [](float x) { return tanhf(x); };
    // auto softmax_ = [](){}; // to be implemented later, it needs a vector of inputs here instead

    switch (activation) {
        case Activation::None: return none_;
        case Activation::ReLu: return relu_;
        case Activation::Sigmoid: return sigmoid_;
        case Activation::Tanh: return tanh_;
        default: assert(false && "Not implemented");
    }
}
