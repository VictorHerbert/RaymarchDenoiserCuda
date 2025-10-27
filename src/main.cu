#include "window.cuh"
#include "test.cuh"

int main(int argc, char* argv[]){
    if (argc > 1) {
        if (strcmp(argv[1], "-gui") == 0) {
            window();
        } else if (strcmp(argv[1], "-t") == 0) {
            test();
        }
    }
    return 0;
}