#ifndef VULKAN_TUTOR_UTILS_H
#define VULKAN_TUTOR_UTILS_H

#include <iostream>
#include <vector>

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
constexpr bool enableDebugMessenger = false;
constexpr bool exitWithStackTrace = false;
#else
constexpr bool enableValidationLayers = true;
constexpr bool enableDebugMessenger = true;
constexpr bool exitWithStackTrace = true;
#endif

std::ostream &print_time(std::ostream &out);
std::vector<char> readFile(const char *fileName);

#endif //VULKAN_TUTOR_UTILS_Hs
