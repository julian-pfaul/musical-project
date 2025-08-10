#include <MidiFile.h>
#include <print>
#include <filesystem>
#include <system_error>

namespace stdfs = std::filesystem;

int main(int argc, char** argv) {
    for (int index = 0; index < argc; ++index) {
        std::println("{}: {}", index, argv[index]);
    }

    std::error_code error;

    for (int index = 1; index < argc; ++index) {
        auto absolute_path = stdfs::absolute(stdfs::path(argv[index]), error);

        if (error) {
            std::println("error: {}", error.message());
            std::exit(1);
        }

        std::println("path {}: {}", index, absolute_path.native());


    }
}
