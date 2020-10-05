#include "utils.h"

#include <iomanip>
#include <ctime>
#include <stdexcept>
#include <fstream>

std::ostream &print_time(std::ostream &out) {
    std::timespec ts;
    if (std::timespec_get(&ts, TIME_UTC) != TIME_UTC) {
        throw std::runtime_error("Can not get current time");
    }
    std::tm* now = std::localtime(&ts.tv_sec);
    return out << '[' << std::setfill('0')
               << (now->tm_year + 1900) << '-'
               << std::setw(2) << (now->tm_mon + 1) << '-'
               << std::setw(2) << now->tm_mday << ' '
               << std::setw(2) << now->tm_hour << ':'
               << std::setw(2) << now->tm_min << ':'
               << std::setw(2) << now->tm_sec << '.'
               << std::setw(9) << ts.tv_nsec
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
