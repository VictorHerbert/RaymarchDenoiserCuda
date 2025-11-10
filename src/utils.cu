#include "utils.cuh"

#include "third_party/helper_math.h"



void printGPUProperties(){
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0f << " KB" << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024.0f << " KB" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
}