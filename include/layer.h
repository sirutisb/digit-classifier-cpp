#pragma once
#include <cstddef>
#include <cmath>
#include <vector>
#include <functional>

enum class Activation {
    None,
    ReLu,
    Sigmoid,
    Tanh,
    Softmax
};

struct Layer {
    const size_t inputSize;
    const size_t outputSize;
    std::vector<float> weights;
    std::vector<float> biases;
    std::function<void(std::vector<float>&)> fn;

    Layer(size_t in, size_t out, const std::function<void(std::vector<float>&)>& activation)
    : inputSize(in)
    , outputSize(out)
    , weights(in * out)
    , biases(out, 0.0f)
    , fn(activation)
    {
        float epsilon = 1.0f / sqrtf((float)inputSize);
        for (size_t i = 0; i < weights.size(); ++i) {
            float r = (float)rand() / (float)RAND_MAX;
            weights[i] = (r - 0.5f) * epsilon;
        }
    }
};

struct LayerConfig {
    size_t size;
    Activation activation;
};
