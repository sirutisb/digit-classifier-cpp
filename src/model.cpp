#include "model.h"
#include "activations.h"
#include "image.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <print>
#include "loader.h"

Model::Model(const std::initializer_list<LayerConfig>& config) {
    if (config.size() <= 1) return;
    size_t layer_count = config.size();
    weights_.reserve(layer_count - 1);
    biases_.reserve(layer_count - 1);
    layerSizes_.reserve(layer_count);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto xavier_init = [&](size_t input_size) {
        return std::normal_distribution<float>(0.0, sqrtf(1.0f / static_cast<float>(input_size))); // Standard Xavier for Sigmoid
    };

    auto he_init = [&](size_t input_size) {
        // Variance = 2.0 / input_size
        return std::normal_distribution<float>(0.0, std::sqrt(2.0f / static_cast<float>(input_size))); // for relu
    };

    // Init weights
    for (auto in = config.begin(); in != config.end();) {
        layerSizes_.push_back(in->size);
        auto out = std::next(in);
        if (out == config.end()) break;
        size_t weight_count = in->size * out->size;

        auto& ws = weights_.emplace_back();
        ws.reserve(weight_count);
        // auto w_dist = xavier_init(in->size);
        auto w_dist = he_init(in->size);
        for (size_t i = 0; i < weight_count; ++i) ws.push_back(w_dist(gen));
        biases_.emplace_back(out->size, 0); // set output size and fill with 0's

        std::println("Created Layer | Layer Size: {} | Weights: {} | Biases: {}", layerSizes_.back(), ws.size(), biases_.back().size());

        in = out;
    }

    std::println("w: {}, b: {}, L: {}", weights_.size(), biases_.size(), layerSizes_.size());

    // load pretrained as a test
    // auto w1 = load_floats("../test/0_weight.bin", weights_[0].size());
    // auto w2 = load_floats("../test/2_weight.bin", weights_[1].size());
    // auto w3 = load_floats("../test/4_weight.bin", weights_[2].size());
    // auto b1 = load_floats("../test/0_bias.bin", biases_[0].size());
    // auto b2 = load_floats("../test/2_bias.bin", biases_[1].size());
    // auto b3 = load_floats("../test/4_bias.bin", biases_[2].size());

    // weights_[0] = w1;
    // weights_[1] = w2;
    // weights_[2] = w3;

    // biases_[0] = b1;
    // biases_[1] = b2;
    // biases_[2] = b3;
}

std::vector<TrainHistory> Model::fit(const std::vector<LabeledImage>& train, int epochs, int batch_size, float learning_rate) {
    std::vector<TrainHistory> history;
    history.reserve(epochs);

    size_t num_layers = layerSizes_.size();

    // Activations
    std::vector<std::vector<float>> a(num_layers);
    for (size_t i = 0; i < num_layers; ++i) a[i].resize(layerSizes_[i]);

    // Deltas store the error terms. d[i] correspond to error at layer i + 1
    // d is alligned with weights: d[0] is error for target of weights_[0]
    std::vector<std::vector<float>> d(num_layers - 1);
    for (size_t i = 0; i < d.size(); ++i) d[i].resize(a[i + 1].size()); // skip the input layer

    // Gradient Accumulators
    std::vector<std::vector<float>> w_grad(weights_.size());
    std::vector<std::vector<float>> b_grad(biases_.size());

    for (size_t i = 0; i < w_grad.size(); ++i) {
        w_grad[i].assign(weights_[i].size(), 0.0f);
        b_grad[i].assign(biases_[i].size(), 0.0f);
    }

    float prev_acc = 0.0f;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;
        int items_processed = 0;
        for (size_t idx = 0; idx < train.size(); ++idx) {
            const auto& sample = train[idx];
            items_processed++;

            assert(a[0].size() == IMAGE_SIZE && "first activations should be as big as the image");
            std::copy(sample.image, sample.image + IMAGE_SIZE, a[0].data());

            for (int l = 0; l < weights_.size(); ++l) {
                const auto& w = weights_[l];
                const auto& b = biases_[l];
                const auto& input = a[l];
                auto& output = a[l + 1];

                size_t input_size = input.size();
                size_t output_size = output.size();

                for (size_t j = 0; j < output_size; ++j) {
                    float z = b[j];
                    for (size_t k = 0; k < input_size; ++k) {
                        z += w[j * input_size + k] * input[k];
                    }
                    output[j] = sigmoid(z);
                }
            }

            // compute loss
            const auto& output = a.back();
            float sample_loss = 0.0f;
            uint8_t pred_digit = 0;
            float max_val = output[0];

            assert(output.size() == d.back().size());

            for (size_t j = 0; j < output.size(); ++j) {
                float target = (j == sample.label) ? 1.0f : 0.0f;
                float error = output[j] - target; // (a - y)
                sample_loss += error * error;

                // Compute output layer delta
                float dC_da = error;
                float da_dz = output[j] * (1.0f - output[j]);
                d.back()[j] = dC_da * da_dz;

                if (output[j] > max_val) {
                    max_val = output[j];
                    pred_digit = j;
                }
            }
            epoch_loss += sample_loss;
            if (pred_digit == sample.label) correct_predictions++;

            // back prop, finding rest of deltas (i swapped j k here)
            for (int i = (int)d.size() - 2; i >= 0; --i) {
                const auto& w_next = weights_[i + 1];
                const auto& a_curr = a[i + 1]; // because they are not alligned (and input layer misaligns them)
                const auto& d_next = d[i + 1];
                auto& d_curr = d[i];

                for (size_t k = 0; k < d_curr.size(); ++k) {
                    float error_sum = 0.0f;
                    for (size_t j = 0; j < d_next.size(); ++j) {
                        float dz_da = w_next[j * d_curr.size() + k];
                        error_sum += d_next[j] * dz_da;
                    }
                    float da_dz = a_curr[k] * (1.0f - a_curr[k]);
                    d_curr[k] = error_sum * da_dz;
                }
            }

            // accumulate gradients over the batch
            assert(weights_.size() + 1 == a.size() && "d must be 1 smaller than activations");
            for (int l = 0; l < weights_.size(); ++l) {
                size_t input_cols = a[l].size();
                size_t output_rows = d[l].size();

                for (int j = 0; j < output_rows; ++j) {
                    float delta = d[l][j];
                    b_grad[l][j] += delta;

                    for (int k = 0; k < input_cols; ++k) {
                        float dz_dw = a[l][k];
                        float grad_w = delta * dz_dw; // gradient for current w for j in next layer from k in prev layer
                        w_grad[l][j * input_cols + k] += grad_w;
                    }
                }
            }

            if (items_processed >= batch_size || idx == train.size() - 1) {
                float scaler = learning_rate / static_cast<float>(items_processed);

                for (size_t l = 0; l < weights_.size(); ++l) {
                    float w_size = weights_[l].size();
                    float b_size = biases_[l].size();

                    for (size_t w = 0; w < w_size; ++w) {
                        weights_[l][w] -= w_grad[l][w] * scaler;
                        w_grad[l][w] = 0.0f;
                    }
                    for (size_t b = 0; b < b_size; ++b) {
                        biases_[l][b] -= b_grad[l][b] * scaler;
                        b_grad[l][b] = 0.0f;
                    }
                }
                items_processed = 0;
            }
        }

        float acc = 100.0f * static_cast<float>(correct_predictions) / train.size();

        // Track the best accuracy we ever seen
        static float best_acc = 0.0f;
        static int no_improve = 0;

        // Check if we improved by at least a small amount
        float min_improvement = 0.3f;
        if (acc > (best_acc + min_improvement)) {
            best_acc = acc;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= 3) {
                learning_rate /= 2.0f;
                no_improve = 0;
                std::println("Plateau detected. Halving LR. New LR: {:.6f}", learning_rate);

                if (learning_rate < 0.01f) {
                    std::println("No improvement after many halvings.");
                    return history;
                }
            }
        }

        std::println("Epoch: {} | Loss: {:.4f} | Acc: {:.2f}%", epoch, epoch_loss / train.size(), acc);
        history.push_back({epoch, epoch_loss, acc});
    }
    return history;
}



uint8_t Model::predict(const Image& im) const {
    auto result = forwardPass(im);
    assert(result.size() == 10 && "should be 10");
    uint8_t maxDigit = 0;
    float maxScore = result[0];
    for (size_t i = 1; i < result.size(); ++i) {
        if (result[i] > maxScore) {
            maxScore = result[i];
            maxDigit = i;
        }
    }
    return maxDigit;
}

void Model::evaluate(const std::vector<LabeledImage>& test) const {
    int correct = 0;
    std::vector<std::vector<int>> cm(10, std::vector<int>(10, 0));
    for (const auto& sample : test) {
        uint8_t pred = predict(sample.image);
        uint8_t label = sample.label;
        cm[pred][label]++;
        if (pred == label) correct++;
    }
    size_t total = test.size();
    std::println("{}/{} ({}%)", correct, total, 100.f * (float)correct / (float)total);
}

std::vector<float> Model::forwardPass(const Image& image) const {
    size_t layers = layerSizes_.size();
    std::vector<float> a(layerSizes_[0]);
    assert(layerSizes_[0] == IMAGE_SIZE && "first layer should have input as image shape");
    std::copy(image, image + IMAGE_SIZE, a.data());

    for (int l = 1; l < layers; ++l) {
        std::vector<float> next_a(layerSizes_[l]);
        for (size_t j = 0; j < next_a.size(); ++j) {
            float z = biases_[l - 1][j];
            for (size_t k = 0; k < a.size(); ++k) {
                z += weights_[l - 1][j * a.size() + k] * a[k];
            }
            next_a[j] = sigmoid(z);
        }
        a = std::move(next_a);
    }
    return a;
}