#include "image.h"
#include <string>
#include <vector>

std::vector<LabeledImage> loadImages(const std::string& imPath, const std::string& lbPath, uint32_t maxSamples = 0);
