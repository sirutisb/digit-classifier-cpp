#include "model.h"
#include "layer.h"
#include <algorithm>
#include <cmath>
#include <cassert>

Model::Model(const std::initializer_list<LayerConfig>& config, float lr) : lr_(lr) {
    layers_.reserve(config.size() - 1);
    for (auto it = config.begin();;) {
        auto next = std::next(it);
        if (next == config.end()) break;
        auto fn = get_activation(next->activation);
        layers_.emplace_back(it->size, next->size, fn);
        it = next;
    }
}

void Model::fit(const std::vector<LabeledImage>& data, int epochs) {
    for (int e = 0; e < epochs; ++e) {
        for (const auto& sample : data) {
            // 1. compute forward pass on the data and store the intermediate values
            // 2. compute error at the output layer to the true label
            // 3. backpropogate the error through the hidden layers
            for (int i = layers_.size() - 1; i >= 0; --i) {

            }
        }
    }
}

uint8_t Model::predict(const Image& image) const {
    std::vector<float> activations(&image[0][0], &image[0][0] + 784);

    for (const Layer& layer : layers_) {
        std::vector<float> nextActivations(layer.outputSize, 0.0f);
        for (size_t i = 0; i < layer.outputSize; ++i) {
            for (size_t j = 0; j < layer.inputSize; ++j) {
                nextActivations[i] += activations[j] * layer.weights[i * layer.inputSize + j];
            }
            nextActivations[i] += layer.biases[i];
        }
        layer.fn(nextActivations);
        activations = std::move(nextActivations);
    }

    float max_pred = activations[0];
    uint8_t pred_digit = 0;
    for (size_t i = 1; i < activations.size(); ++i) {
        float y = activations[i];
        if (y > max_pred) {
            max_pred = y;
            pred_digit = (uint8_t)i;
        }
    }
    return pred_digit;
}

std::function<void(std::vector<float>&)> Model::get_activation(Activation activation) {
    using Vec = std::vector<float>;
    auto apply_inplace = [](Vec& v, auto&& op) {
        std::transform(v.begin(), v.end(), v.begin(), [&](float x) { return op(x); });
    };

    auto none_ = [](Vec&) {};
    auto relu_ = [&](Vec& v) { apply_inplace(v, [](float x) { return std::max(x, 0.0f); }); };
    auto sigmoid_ = [&](Vec& v) { apply_inplace(v, [](float x) { return 1.0f / (1.0f + expf(-x)); }); };
    auto tanh_ = [&](Vec& v) { apply_inplace(v, [](float x) { return tanhf(x); }); };

    auto softmax_ = [](Vec& v) {
        const float max_val = *std::max_element(v.begin(), v.end());
        float sum = 0.0f;
        for (float x : v) sum += expf(x - max_val); // offset by max_val such that we don't get numbers too high
        for (float& x : v) x = expf(x - max_val) / sum;
    };

    switch (activation) {
        case Activation::None:      return none_;
        case Activation::ReLu:      return relu_;
        case Activation::Sigmoid:   return sigmoid_;
        case Activation::Tanh:      return tanh_;
        case Activation::Softmax:   return softmax_;
        default: assert(false && "Not implemented");
    }
}
