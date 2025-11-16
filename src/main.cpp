#include "test.h"
#include <cstring>
#include <iostream>

void print_help(char *program_name){
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -t [label]   Run tests (all or specific label)\n";
    std::cout << "  -h           Show this help message\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2){
        print_help(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0) {
            const char* label = nullptr;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                label = argv[i + 1];
                ++i;
            }
            if (label) {
                test(label);
            } else {
                test();
            }
        } else if (strcmp(argv[i], "-h") == 0){
            print_help(argv[0]);
            return 1;
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
        }
    }

    return 0;
}
