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

SKIP(IMAGE){
    Image image3(IMAGE_SAMPLE_PATH, 3);
    image3.save(OUTPUT_PATH + "image_open_save3.png");

    Image image4(IMAGE_SAMPLE_PATH, 4);
    image4.save(OUTPUT_PATH + "image_open_save4.png");
}


int2 shape =  {1920, 1080};
CudaVector<uchar4> in(totalSize(shape));
CudaVector<uchar4> out(totalSize(shape));

TEST(FILTER_BASELINE){

    dim3 blockSize(16,16);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    filterKernelBaseline<<<gridSize, blockSize, 49152>>>(
        {.shape=shape, .render=in.data(), .denoised=out.data()},
        {.type=FilterParams::AVERAGE, .depth=1, .radius=2});

    cudaDeviceSynchronize();
}


TEST(FILTER_TILED){
    dim3 blockSize(16,16);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    filterKernelTiled<<<gridSize, blockSize, 30*1024>>>(
        {.shape=shape, .render=in.data(), .denoised=out.data()},
        {.type=FilterParams::AVERAGE, .depth=1, .radius=2});

    cudaDeviceSynchronize();
}