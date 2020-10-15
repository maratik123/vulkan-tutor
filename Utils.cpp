#include "Utils.h"

#include <ostream>
#include <fstream>
#include <ctime>
#include <iomanip>

std::ostream &print_time(std::ostream &out) {
    std::timespec ts{};
    std::tm localtime_result{};

    if (std::timespec_get(&ts, TIME_UTC) != TIME_UTC) {
        throw std::runtime_error("Can not get current time");
    }

    if (::localtime_r(&ts.tv_sec, &localtime_result) != &localtime_result) {
        throw std::runtime_error("Can not split current time");
    }

    return out << '['
               << std::put_time(&localtime_result, "%F %T")
               << '.' << std::setfill('0') << std::setw(9) << ts.tv_nsec
               << std::put_time(&localtime_result, "%z")
               << "] ";
}

std::vector<char> readFile(const char *fileName) {
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open file: ") + fileName);
    }

    std::size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    return buffer;
}
