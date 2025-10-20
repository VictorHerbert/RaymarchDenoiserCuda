#include "image.cuh"
#include "filter.cuh"
#include "test.cuh"

#include <vector>
#include <functional>

FuncVector registered_funcs;

void test() {
    if(registered_funcs.empty()){
        std::cout << "No tests found" << std::endl;
        return;
    }
    for (auto& [name, func] : registered_funcs) {
        std::cout << "TEST " << name << "\t: ";
        try {
            func();
            std::cout << "passed" << std::endl;
        } catch (...) {
            std::cout << "failed" << std::endl;
        }
    }
}

TEST(image){
    Image image("render/cornell/1/render.png");
    image.save("build/output/image.png");
}

TEST(filter_cpu){
    Image render_img("render/cornell/1/render.png");
    Image albedo_img("render/cornell/1/albedo.png");
    Image normal_img("render/cornell/1/normal.png");
    
    int2 shape = render_img.shape;

    Image denoised_img(shape);

    waveletfilterCPU(
        {shape, render_img.vecBuffer.data(), normal_img.vecBuffer.data(), albedo_img.vecBuffer.data(), denoised_img.vecBuffer.data()},
        {5, 0.5f, 0.5f, 0.5f, 0.3f}
    );    

    denoised_img.save("build/output/filter_cpu.png");
}

TEST(filter_gpu){
    Image render_img("render/cornell/1/render.png");
    Image albedo_img("render/cornell/1/albedo.png");
    Image normal_img("render/cornell/1/normal.png");
    
    int2 shape = render_img.shape;

    Image denoised_img(shape);

    CudaVector<uchar3> render(render_img.vecBuffer);
    CudaVector<uchar3> albedo(albedo_img.vecBuffer);
    CudaVector<uchar3> normal(normal_img.vecBuffer);
    CudaVector<uchar3> denoised(totalSize(shape));

    waveletfilterGPU(
        {shape, render.data(), normal.data(), albedo.data(), denoised.data()},
        {5, 0.5f, 0.5f, 0.5f, 0.3f}
    ); 
    denoised.copyTo(denoised_img.vecBuffer);
    
 
    denoised_img.save("build/output/filter_gpu.png");
}