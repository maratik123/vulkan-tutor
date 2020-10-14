#ifndef VULKAN_TUTOR_IMAGE_H
#define VULKAN_TUTOR_IMAGE_H

#include <cstddef>

struct Image {
    unsigned char *pixels;
    std::size_t imageSize;
    int texWidth;
    int texHeight;
    int texChannels;

    explicit Image(const char *fileName);
    ~Image();
};


#endif //VULKAN_TUTOR_IMAGE_H
