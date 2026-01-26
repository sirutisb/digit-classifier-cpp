#include "network.h"
#include "activations.h"
#include <print>

std::vector<float> forwardPass(
const Image& image,
std::vector<float>& w1, std::vector<float>& w2, std::vector<float>& w3,
std::vector<float>& b1, std::vector<float>& b2, std::vector<float>& b3) {
    std::vector<float> activations(image, image + IMAGE_SIZE);
    std::vector<float> next;

    // 1st hidden layer
    next = std::vector<float>(16, 0);
    for (int neuron = 0; neuron < next.size(); ++neuron) {
        float z = b1[neuron];
        for (int i = 0; i < activations.size(); ++i) {
            z += w1[neuron * activations.size() + i] * activations[i];
        }
        next[neuron] = z;
    }
    activations = std::move(next);
    sigmoid_all(activations);

    // 2nd hidden layer
    next = std::vector<float>(16, 0);
    for (int neuron = 0; neuron < next.size(); ++neuron) {
        float z = b2[neuron];
        for (int i = 0; i < activations.size(); ++i) {
            z += w2[neuron * activations.size() + i] * activations[i];
        }
        next[neuron] = z;
    }
    activations = std::move(next);
    sigmoid_all(activations);

    // output layer
    next = std::vector<float>(10, 0);
    for (int neuron = 0; neuron < next.size(); ++neuron) {
        float z = b3[neuron];
        for (int i = 0; i < activations.size(); ++i) {
            z += w3[neuron * activations.size() + i] * activations[i];
        }
        next[neuron] = z;
    }
    activations = std::move(next);
    sigmoid_all(activations);

    return activations;
}

void run_model_iter(const std::vector<LabeledImage>& train, int batch_size, float learning_rate, int epoch,
std::vector<float>& w1, std::vector<float>& w2, std::vector<float>& w3,
std::vector<float>& b1, std::vector<float>& b2, std::vector<float>& b3);

void train_model(
const std::vector<LabeledImage>& train, int epochs, int batch_size, float learning_rate,
std::vector<float>& w1, std::vector<float>& w2, std::vector<float>& w3,
std::vector<float>& b1, std::vector<float>& b2, std::vector<float>& b3) {

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        run_model_iter(train, batch_size, learning_rate, epoch, w1, w2, w3, b1, b2, b3);
    }
}

void run_model_iter(
const std::vector<LabeledImage>& train, int batch_size, float learning_rate, int epoch,
std::vector<float>& w1, std::vector<float>& w2, std::vector<float>& w3,
std::vector<float>& b1, std::vector<float>& b2, std::vector<float>& b3
) {
    float sampleLoss = 0.0f;
    std::vector<float> w1_desired(w1.size()), w2_desired(w2.size()), w3_desired(w3.size());
    std::vector<float> b1_desired(b1.size()), b2_desired(b2.size()), b3_desired(b3.size());
    for (size_t idx = 0; idx < train.size(); ++idx) {
        const auto& sample = train[idx];
        const Image& image = sample.image;

        std::vector<float> a0(image, image + IMAGE_SIZE);
        std::vector<float> a1(16);
        std::vector<float> a2(16);
        std::vector<float> a3(10);

        // Computing Forward Pass

        // 1st hidden layer (16)
        for (int j = 0; j < a1.size(); ++j) {
            float z = b1[j];
            for (int k = 0; k < a0.size(); ++k) {
                z += w1[j * a0.size() + k] * a0[k];
            }
            a1[j] = sigmoid(z);
        }

        // 2nd hidden layer (16)
        for (int j = 0; j < a2.size(); ++j) {
            float z = b2[j];
            for (int k = 0; k < a1.size(); ++k) {
                z += w2[j * a1.size() + k] * a1[k];
            }
            a2[j] = sigmoid(z);
        }

        // output layer (10)
        for (int j = 0; j < a3.size(); ++j) {
            float z = b3[j];
            for (int k = 0; k < a2.size(); ++k) {
                z += w3[j * a2.size() + k] * a2[k];
            }
            a3[j] = sigmoid(z);
        }

        // get loss to use outside
        for (int j = 0; j < a3.size(); ++j) {
            float error = a3[j] - (j == sample.label ? 1.0f : 0.0f);
            sampleLoss += error * error;
        }

        // Compute Delta
        // d = dC/da * da/dz

        // delta 3
        std::vector<float> d3(a3.size());
        for (int j = 0; j < a3.size(); ++j) {
            float dC_da = a3[j] - (j == sample.label ? 1.0f : 0.0f);
            float da_dz = a3[j] * (1.0f - a3[j]);
            d3[j] = dC_da * da_dz;
        }
        // delta 2
        std::vector<float> d2(a2.size());
        for (int k = 0; k < a2.size(); ++k) {
            for (int j = 0; j < a3.size(); ++j) {
                float dz_da = w3[j * a2.size() + k]; // dz from current layer to ak from l-1
                d2[k] += d3[j] * dz_da;
            }
            float da_dz = a2[k] * (1.0f - a2[k]);
            d2[k] *= da_dz;
        }
        // delta 1
        std::vector<float> d1(a1.size());
        for (int k = 0; k < a1.size(); ++k) {
            for (int j = 0; j < a2.size(); ++j) {
                float dz_da = w2[j * a1.size() + k]; // dz from current layer to ak from l-1
                d1[k] += d2[j] * dz_da;
            }
            float da_dz = a1[k] * (1.0f - a1[k]);
            d1[k] *= da_dz;
        }

        for (int j = 0; j < a3.size(); ++j) {
            for (int k = 0; k < a2.size(); ++k) {
                float dz_dw = a2[k];
                float grad_w = d3[j] * dz_dw; // gradient for current w3 for j in l3 from k in l2
                w3_desired[j * a2.size() + k] -= grad_w;
            }
            b3_desired[j] -= d3[j];
        }


        // descent for delta 2 now we can treat l2 as j
        for (int j = 0; j < a2.size(); ++j) {
            for (int k = 0; k < a1.size(); ++k) {
                float dz_dw = a1[k];
                float grad_w = d2[j] * dz_dw; // gradient for current w2 for j in l2 from k in l1
                w2_desired[j * a1.size() + k] -= grad_w;
            }
            b2_desired[j] -= d2[j];
        }

        // descent for delta 1 now we can treat l1 as j
        for (int j = 0; j < a1.size(); ++j) {
            for (int k = 0; k < a0.size(); ++k) {
                float dz_dw = a0[k];
                float grad_w = d1[j] * dz_dw; // gradient for current w1 for j in l1 from k in l0
                w1_desired[j * a0.size() + k] -= grad_w;
            }
            b1_desired[j] -= d1[j];
        }

        if (idx % batch_size == 0 || idx == train.size() - 1) {
            for (size_t i = 0; i < w1.size(); ++i) { w1[i] += w1_desired[i] / batch_size; w1_desired[i] = 0.0f; }
            for (size_t i = 0; i < w2.size(); ++i) { w2[i] += w2_desired[i] / batch_size; w2_desired[i] = 0.0f; }
            for (size_t i = 0; i < w3.size(); ++i) { w3[i] += w3_desired[i] / batch_size; w3_desired[i] = 0.0f; }
            for (size_t i = 0; i < b1.size(); ++i) { b1[i] += b1_desired[i] / batch_size; b1_desired[i] = 0.0f; }
            for (size_t i = 0; i < b2.size(); ++i) { b2[i] += b2_desired[i] / batch_size; b2_desired[i] = 0.0f; }
            for (size_t i = 0; i < b3.size(); ++i) { b3[i] += b3_desired[i] / batch_size; b3_desired[i] = 0.0f; }
            // get accuracy:
            int batch_num = idx / batch_size;
            std::println("epoch: {} | Batch {} | Avg. Loss: {}", epoch, batch_num, sampleLoss / batch_size);
            sampleLoss = 0.0f;
        }
    }
}