#include "image.h"
#include <string>
#include <vector>

std::vector<LabeledImage> loadImages(const std::string& imPath, const std::string& lbPath, uint32_t maxSamples = 0);

// path to binary file containing an array of float32 values. Size given in number of floats
std::vector<float> load_floats(const std::string& path, int size);


std::pair<std::vector<LabeledImage>, std::vector<LabeledImage>> load_train_test(size_t count = 0);