#include "image.h"
#include "filter.cuh"
#include "test.h"

#include "third_party/stb_image.h"

#include <vector>
#include <cassert>
#include <regex>


const std::string OUTPUT_PATH =  "test/";
const std::string IMAGE_SAMPLE_PATH =  "render/sponza/render/1.png";

FuncVector registered_funcs;

void test(std::string wildcard) {
    printf("----------------------------------------------------------\n");
    printf("%d available tests: ", registered_funcs.size());
    for (auto& [name, func] : registered_funcs)
        printf("%s ", name.c_str());
    printf("\n----------------------------------------------------------\n");

    std::regex base_regex(wildcard);

    for (auto& [name, func] : registered_funcs) {
        if (!std::regex_match(name, base_regex)) {
            continue;
        }

        try {
            printf("TEST %s:\n", name.c_str());
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(end - start).count();

            printf("Passed with %.3f ms\n", frameTime);
        }
        catch (const std::runtime_error& e) {
            printf("Fail with %s\n", e.what());
        }
        catch (...) {
            printf("Failed\n");
        }
        printf("----------------------------------------------------------\n");
    }
}


SKIP(DEVICE_STATS){
    printGPUProperties();
}

TEST(IMAGE){
    Image image3(IMAGE_SAMPLE_PATH, 3);
    image3.save(OUTPUT_PATH + "image_open_save3.png");

    Image image4(IMAGE_SAMPLE_PATH, 4);
    image4.save(OUTPUT_PATH + "image_open_save4.png");
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

SKIP(FILTER_UCHAR4){
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

SKIP(FILTER_BASELINE4){
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


SKIP(FILTER_BASELINE3){
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