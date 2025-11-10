#include "image.cuh"
#include "filter.cuh"
#include "test.cuh"

#include <vector>
#include <cassert>


FuncVector registered_funcs;
const std::string OUTPUT_PATH =  "test/";
const std::string SPONZA_SAMPLE =  "render/sponza/render/1.png";

void test() {
    if(registered_funcs.empty()){
        std::cout << "No tests found" << std::endl;
        return;
    }
    std::cout << "TEST SUITE: " << registered_funcs.size() << " test" << ((registered_funcs.size() == 1) ? "" : "s") <<" found" << std::endl;
    for (auto& [name, func] : registered_funcs) {
        std::cout << "----------------------------------------------------------" << std::endl;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "TEST " << name << ": passed with " << frameTime << " ms" <<std::endl;
        }
        catch (const std::runtime_error& e) {
            std::cout << "TEST " << name << ": failed \nREASON: " << e.what() << std::endl;
        }
        catch (...) {
            std::cout << "TEST " << name << ": failed" << std::endl;
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


TEST(FILTER){
    Image image(SPONZA_SAMPLE);
    int2 shape = image.shape;
    CudaVector<uchar3> in(image.data, totalSize(image.shape));
    CudaVector<uchar3> out(totalSize(image.shape));

    const dim3 blockSize(16,16);
    const bool cacheInput = true;

    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    int byteCount = cacheInput ? totalSize(make_int2(blockSize.x, blockSize.y)) * 25 * sizeof(uchar3) : 0;

    auto start = std::chrono::high_resolution_clock::now();

    filterKernel<<<gridSize, blockSize, byteCount>>>(
        {.shape=shape, .render=in.data(), .denoised=out.data()},
        {.type=FilterParams::AVERAGE, .depth=1, .radius=2, .cacheInput=cacheInput});

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now(); 
    double time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
    std::cout << "SharedMem: " << byteCount << "/" << "49152\t";
    std::cout << time << " ms" << std::endl;

    out.copyTo(image.vecBuffer);
    image.save(OUTPUT_PATH + "filter.png");

}

SKIP(FILTER_SPACE_EXP){
    Image image(SPONZA_SAMPLE);
    int2 shape = image.shape;
    CudaVector<uchar3> in(image.data, totalSize(image.shape));
    CudaVector<uchar3> out(totalSize(image.shape));

    CPUVector<dim3> blockSizes = {{8,8}, {16,16}};
    CPUVector<bool> cacheInputs = {false, true};
    
    for(dim3 blockSize : blockSizes){
        for(bool cacheInput : cacheInputs){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            int byteCount = cacheInput ? totalSize(make_int2(blockSize.x, blockSize.y)) * 25 * sizeof(uchar3) : 0;

            auto start = std::chrono::high_resolution_clock::now();

            filterKernel<<<gridSize, blockSize, byteCount>>>(
                {.shape=shape, .render=in.data(), .denoised=out.data()},
                {.type=FilterParams::AVERAGE, .depth=1, .radius=2, .cacheInput=cacheInput});

            cudaDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now(); 
            double time = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
            std::cout << "SharedMem: " << byteCount << "/" << "49152\t";
            std::cout << time << " ms" << std::endl;
        }
    }

}