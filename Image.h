#ifndef VULKAN_TUTOR_IMAGE_H
#define VULKAN_TUTOR_IMAGE_H

#include <cstddef>

struct Image {
    int texWidth;
    int texHeight;
    int texChannels;
    unsigned char *pixels;
    std::size_t imageSize;

    Image(const char *fileName);
    ~Image();
};


#endif //VULKAN_TUTOR_IMAGE_H
