#include "utils.h"

#include <stdexcept>
#include <fstream>

#include "absl/time/time.h"
#include "absl/time/clock.h"

namespace {
    const absl::TimeZone localTimeZone = absl::LocalTimeZone();
}

std::ostream &print_time(std::ostream &out) {
    return out << '[' << absl::FormatTime(absl::Now(), localTimeZone) << "] ";
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
