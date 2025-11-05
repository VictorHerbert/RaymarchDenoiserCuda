#include "image.cuh"
#include "filter.cuh"
#include "test.cuh"

#include <vector>


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

SKIP(image_open){
    Image image("render/cornell/1/render.png");
}

SKIP(image_open_save){
    Image image("render/cornell/1/render.png");
    image.save(OUTPUT_PATH + "image.png");
}

TEST(FILTER_AVG){
    Image image(SPONZA_SAMPLE);
    int2 shape = image.shape;
    CudaVector<uchar3> in(image.data, totalSize(image.shape));
    CudaVector<uchar3> out(totalSize(image.shape));
    
    const dim3 blockSize(16, 16);
    dim3 gridSize((shape.x + 15) / 16, (shape.y + 15) / 16);

    filterKernel<<<gridSize, blockSize>>>(
        {.shape=shape, .render=in.data(), .denoised=out.data()},
        {.type=FilterParams::AVERAGE, .depth=1});

    out.copyTo(image.vecBuffer);
    image.save(OUTPUT_PATH + "filter_avg.png");
}