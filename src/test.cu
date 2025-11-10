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
    std::cout << registered_funcs.size() << " tests found" << std::endl;
    for (auto& [name, func] : registered_funcs) {
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



Image image(SPONZA_SAMPLE);
int2 shape = image.shape;
CudaVector<uchar3> in(image.data, totalSize(image.shape));
CudaVector<uchar3> out(totalSize(image.shape));

TEST(FILTER_SPACE_EXP){
    CPUVector<dim3> blockSizes = {{8,8}, {16,16}};
    CPUVector<bool> cacheInputs = {false, true};
    
    std::cout << "----------------------------------------------------------" << std::endl;
    for(dim3 blockSize : blockSizes){
        for(bool cacheInput : cacheInputs){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            int byteCount = cacheInput ? totalSize(make_int2(blockSize.x, blockSize.y)) * 25 * sizeof(uchar3) : 0;

            std::cout << "BlockShape: (" << blockSize.x << "," << blockSize.y << ") \t";
            std::cout << "SharedMem: " << byteCount << "/" << 49152 << std::endl;

            filterKernel<<<gridSize, blockSize, byteCount>>>(
                {.shape=shape, .render=in.data(), .denoised=out.data()},
                {.type=FilterParams::AVERAGE, .depth=1, .radius=2, .cacheInput=cacheInput});

            cudaDeviceSynchronize();
        }
    }
    std::cout << "----------------------------------------------------------" << std::endl;
}