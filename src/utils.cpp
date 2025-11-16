#include "utils.h"

#include <cuda_runtime.h>

void printGPUProperties(){
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0f << " KB" << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024.0f << " KB\n" << std::endl;
}