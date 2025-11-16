#include "image.cuh"
#include "filter.cuh"
#include "test.cuh"

#include <vector>
#include <cassert>

#include "third_party/stb_image.h"

FuncVector registered_funcs;
const std::string OUTPUT_PATH =  "test/";
const std::string SPONZA_SAMPLE =  "render/sponza/render/1.png";

void test() {
    if(registered_funcs.empty()){
        std::cout << "No tests found" << std::endl;
        return;
    }
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "TEST SUITE: " << registered_funcs.size() << " test" << ((registered_funcs.size() == 1) ? "" : "s") <<" found" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    for (auto& [name, func] : registered_funcs) {
        
        try {
            std::cout << "TEST " << name << ":" <<std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "Passed with " << frameTime << " ms" <<std::endl;
        }
        catch (const std::runtime_error& e) {
            std::cout << "Fail with " << e.what() << std::endl;
        }
        catch (...) {
            std::cout << "Failed" << std::endl;
        }
        std::cout << "----------------------------------------------------------" << std::endl;
    }
}

SKIP(DEVICE_STATS){
    printGPUProperties();
}

SKIP(image_open){
    Image image("render/cornell/1/render.png");
}

SKIP(image_open_save){
    Image image("render/cornell/1/render.png");
    image.save(OUTPUT_PATH + "image.png");
}

SKIP(FILTER_SINGLE_BLOCK){
    int2 shape =  {1920, 1080};
    CudaVector<uchar4> in(totalSize(shape));
    CudaVector<uchar4> out(totalSize(shape));

    const dim3 blockSize(16,16);
    const bool cacheInput = true;

    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    int byteCount = cacheInput ? 18288 : 0;

    auto start = std::chrono::high_resolution_clock::now();

    filterKernel<uchar4><<<1, blockSize, 49152>>>(
        {.shape=shape, .render=in.data(), .denoised=out.data()},
        {.type=FilterParams::AVERAGE, .depth=1, .radius=2, .cacheInput=cacheInput});

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now(); 
    double time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
    std::cout << "SharedMem: " << 49152 << "/" << "49152\t";
    std::cout << time << " ms\n" << std::endl;
}

dim3 reducedGridSize(4,4);

SKIP(FILTER_UCHAR3){
    int2 shape =  {1920, 1080};
    CudaVector<uchar3> in(totalSize(shape));
    CudaVector<uchar3> out(totalSize(shape));

    CPUVector<dim3> blockSizes = {{8,8}, {16,16}};
    CPUVector<bool> cacheInputs = {false, true};

    CPUVector<std::tuple<dim3, bool>> test_vectors = {
        //{{8,8}, false},
        //{{8,8}, true},
        {{16,16}, false},
        {{16,16}, true},
    };

    for(auto& [blockSize, cacheInput] : test_vectors){
        dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

        int byteCount = cacheInput ? 18288 : 0;

        auto start = std::chrono::high_resolution_clock::now();

        filterKernel<uchar3><<<gridSize, blockSize, byteCount>>>(
            {.shape=shape, .render=in.data(), .denoised=out.data()},
            {.type=FilterParams::AVERAGE, .depth=1, .radius=2, .cacheInput=cacheInput});

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now(); 
        double time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
        std::cout << "CacheInput: " << cacheInput << "\t"; 
        std::cout << "SharedMem: " << byteCount << "/" << "49152\t";
        std::cout << time << " ms" << std::endl;

    }
    std::cout << std::endl;
}

TEST(FILTER_UCHAR4){
    int2 shape = {1920, 1080};
    CudaVector<uchar4> in(totalSize(shape));
    CudaVector<uchar4> out(totalSize(shape));

    CPUVector<dim3> blockSizes = {{8,8}, {16,16}};
    CPUVector<bool> cacheInputs = {false, true};

    CPUVector<std::tuple<dim3, bool>> test_vectors = {
        //{{8,8}, false},
        //{{8,8}, true},
        {{16,16}, false},
        {{16,16}, true},
        {{32,8}, false},
        {{32,8}, true},
    };

    for(auto& [blockSize, cacheInput] : test_vectors){
        dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

        int byteCount = cacheInput ? 18288 : 0;

        auto start = std::chrono::high_resolution_clock::now();

        filterKernel<uchar4><<<gridSize, blockSize, byteCount>>>(
            {.shape=shape, .render=in.data(), .denoised=out.data()},
            {.type=FilterParams::AVERAGE, .depth=1, .radius=2, .cacheInput=cacheInput});

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now(); 
        double time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
        std::cout << "CacheInput: " << cacheInput << "\t"; 
        std::cout << "SharedMem: " << byteCount << "/" << "49152\t";
        std::cout << time << " ms" << std::endl;
    }
    std::cout << std::endl;
}

TEST(FILTER_BASELINE4){
    int2 shape = {1920, 1080};
    CudaVector<uchar4> in(totalSize(shape));
    CudaVector<uchar4> out(totalSize(shape));

    CPUVector<dim3> blockSizes = { {16,16}, {32, 8}, {64, 4}};

    for(auto& blockSize : blockSizes){
        dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

        auto start = std::chrono::high_resolution_clock::now();

        filterKernelBaseline<uchar4><<<gridSize, blockSize>>>(
            {.shape=shape, .render=in.data(), .denoised=out.data()},
            {.type=FilterParams::AVERAGE, .depth=1, .radius=2});

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now(); 
        double time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
        std::cout << time << " ms" << std::endl;
    }
    std::cout << std::endl;
}


TEST(FILTER_BASELINE3){
    int2 shape = {1920, 1080};
    CudaVector<uchar3> in(totalSize(shape));
    CudaVector<uchar3> out(totalSize(shape));

    CPUVector<dim3> blockSizes = { {16,16}, {32, 8}};

    for(auto& blockSize : blockSizes){
        dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

        auto start = std::chrono::high_resolution_clock::now();

        filterKernelBaseline<uchar3><<<gridSize, blockSize>>>(
            {.shape=shape, .render=in.data(), .denoised=out.data()},
            {.type=FilterParams::AVERAGE, .depth=1, .radius=2});

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now(); 
        double time = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
        std::cout << time << " ms" << std::endl;
    }
    std::cout << std::endl;
}