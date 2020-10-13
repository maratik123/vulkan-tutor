#include "Image.h"

#include "STBInclude.h"

Image::Image(const char *fileName) {
    pixels = stbi_load(fileName, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    imageSize = texWidth * texHeight * 4;
}

Image::~Image() {
    stbi_image_free(pixels);
}
