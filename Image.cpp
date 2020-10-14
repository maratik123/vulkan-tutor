#include "Image.h"

#include <stdexcept>

#include "STBInclude.h"

Image::Image(const char *fileName) : texWidth{}, texHeight{}, texChannels{} {
    pixels = stbi_load(fileName, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error(std::string("failed to load image: ") + fileName);
    }

    imageSize = texWidth * texHeight * 4;
}

Image::~Image() {
    stbi_image_free(pixels);
}
