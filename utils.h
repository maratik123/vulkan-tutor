#ifndef VULKAN_TUTOR_UTILS_H
#define VULKAN_TUTOR_UTILS_H

#include <iostream>
#include <vector>

#ifdef NDEBUG
constexpr auto enableValidationLayers = false;
constexpr auto enableDebugMessenger = false;
constexpr auto exitWithStackTrace = false;
#else
constexpr auto enableValidationLayers = true;
constexpr auto enableDebugMessenger = true;
constexpr auto exitWithStackTrace = true;
#endif

std::ostream &print_time(std::ostream &out);
std::vector<char> readFile(const char *fileName);

#endif //VULKAN_TUTOR_UTILS_Hs
