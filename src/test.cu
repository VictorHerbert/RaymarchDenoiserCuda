#include "image.cuh"
#include "filter.cuh"
#include "video.cuh"
#include "test.cuh"

#include <vector>


FuncVector registered_funcs;
const std::string OUTPUT_PATH =  "build/output/";

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

SKIP(image){
    Image image("render/cornell/1/render.png");
    image.save(OUTPUT_PATH + "image.png");
}

TEST(filter_cpu){
    Image render_img("render/cornell/1/render.png");
    Image albedo_img("render/cornell/1/albedo.png");
    Image normal_img("render/cornell/1/normal.png");
    
    int2 shape = render_img.shape;

    Image denoised_img(shape);

    waveletfilterCPU(
        {shape, render_img.vecBuffer.data(), normal_img.vecBuffer.data(), albedo_img.vecBuffer.data(), denoised_img.vecBuffer.data()},
        {5, .1f, .1f, .1f, .1f}
    );    

    denoised_img.save(OUTPUT_PATH + "filter_cpu.png");
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
        {5, .1f, .1f, .1f, .1f}
    ); 
    denoised.copyTo(denoised_img.vecBuffer);
    
    denoised_img.save(OUTPUT_PATH + "filter_gpu.png");
}

TEST(video_gpu) {
    decodeVideo("render/sponzavideo/render.avi", [](uchar3* frame, int2 size) {
        std::cout << "Got frame: " << size.x << "x" << size.y << std::endl;
        Image::save(OUTPUT_PATH + "video.png", frame, size);
    });
}
