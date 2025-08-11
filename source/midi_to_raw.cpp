#include <print>
#include <string_view>
#include <system_error>
#include <filesystem>

#include <MidiFile.h>

namespace stdfs = std::filesystem;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::println(std::cerr, "Invalid argument:\nMust provide exactly one file path argument (argc should be two). argc = {}.", argc);
        return 1;
    }

    auto argument = std::string_view(argv[1]);

    if (not argument.length() > 0) {
        std::println(std::cerr, "Invalid argument:\nProvided argument cannot be an empty string. string = \"{}\". length = {}.", argument, argument.length());
        return 1;
    }

    auto error_code = std::error_code();

    auto path = stdfs::absolute(stdfs::path(argument), error_code);

    if (error_code) {
        std::println(std::cerr, "Error: {} (Code = {}).", error_code.message(), error_code.value());
        return 1;
    }

    smf::MidiFile midifile;
    midifile.read(path.native());

    if (not midifile.status()) {
        std::println(std::cerr, "MIDI Error: path = {}. filename = {}. status = {}.", path.native(), midifile.getFilename(), midifile.status());
        return 1;
    }



    midifile.writeHex(std::cout);
}
