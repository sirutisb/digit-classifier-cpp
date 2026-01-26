#include "loader.h"
#include <cstdint>
#include <progress.h>
#include <bit>
#include <fstream>
#include <iostream>
#include <print>
#include <stdexcept>

std::vector<LabeledImage> loadImages(const std::string& imPath, const std::string& lbPath, uint32_t maxSamples) {
    std::ifstream is(imPath, std::ios::binary);
    std::ifstream ls(lbPath, std::ios::binary);
    if (!is.is_open()) throw std::runtime_error("Unable to open file to images");
    if (!ls.is_open()) throw std::runtime_error("Unable to open file to labels");

    uint32_t im_magic, lb_magic;
    uint32_t im_count, lb_count;
    uint32_t im_rows, im_cols;

    is.read(reinterpret_cast<char*>(&im_magic), 4);
    is.read(reinterpret_cast<char*>(&im_count), 4);
    is.read(reinterpret_cast<char*>(&im_rows), 4);
    is.read(reinterpret_cast<char*>(&im_cols), 4);

    ls.read(reinterpret_cast<char*>(&lb_magic), 4);
    ls.read(reinterpret_cast<char*>(&lb_count), 4);

    im_magic = std::byteswap(im_magic);
    lb_magic = std::byteswap(lb_magic);
    im_count = std::byteswap(im_count);
    lb_count = std::byteswap(lb_count);
    im_rows = std::byteswap(im_rows);
    im_cols = std::byteswap(im_cols);

    if (im_magic != 2051) throw std::runtime_error("Images magic doesnt match");
    if (lb_magic != 2049) throw std::runtime_error("Labels magic doesnt match");
    if (im_count != lb_count) throw std::runtime_error("counts do not match");

    if (maxSamples > 0 && maxSamples < im_count) {
        im_count = lb_count = maxSamples;
    } else maxSamples = im_count;

    std::println("Loading {} images from {} of size {}x{}", im_count, imPath, im_rows, im_cols);
    std::vector<LabeledImage> images(im_count);
    size_t n = im_rows * im_cols;
    const size_t chunks = im_count < 100 ? 1 : im_count / 100;
    for (size_t im = 0; im < im_count; ++im) {
        uint8_t d; ls.read(reinterpret_cast<char*>(&d), 1);
        images[im].label = (uint8_t)d;
        for (size_t p = 0; p < n; ++p) {
            uint8_t intensity; is.read(reinterpret_cast<char*>(&intensity), 1);
            images[im].image[p] = static_cast<float>(intensity) / 255.0f;
        }
        if (im % chunks == 0 || im + 1 == im_count) {
            // std::println("Why failed");
            updateProgress(static_cast<float>(im + 1) / static_cast<float>(im_count));
        }
    }
    return images;
}

std::vector<float> load_floats(const std::string& path, int size) {
    std::vector<float> buffer(size);
    std::ifstream file(path, std::ios::binary);

    if (file.is_open()) {
        // Read size * 4 bytes directly into the vector's memory
        file.read(reinterpret_cast<char*>(buffer.data()), size * sizeof(float));
        file.close();
    } else {
        std::println("Error: Could not open {}", path);
        throw std::runtime_error("Cannot find file");
    }
    return buffer;
}

std::pair<std::vector<LabeledImage>, std::vector<LabeledImage>> load_train_test(size_t train_count, size_t test_count) {
    auto train = loadImages(
        "../dataset/train-images.idx3-ubyte",
        "../dataset/train-labels.idx1-ubyte",
        train_count
    );

    auto test = loadImages(
        "../dataset/t10k-images.idx3-ubyte",
        "../dataset/t10k-labels.idx1-ubyte",
        test_count
    );

    return std::make_pair(train, test);
}

void load_pretrained(
    std::vector<float>& w1, std::vector<float>& w2, std::vector<float>& w3,
    std::vector<float>& b1, std::vector<float>& b2, std::vector<float>& b3
) {
    w1 = load_floats("../test/0_weight.bin", w1.size());
    w2 = load_floats("../test/2_weight.bin", w2.size());
    w3 = load_floats("../test/4_weight.bin", w3.size());
    b1 = load_floats("../test/0_bias.bin", b1.size());
    b2 = load_floats("../test/2_bias.bin", b2.size());
    b3 = load_floats("../test/4_bias.bin", b3.size());
}