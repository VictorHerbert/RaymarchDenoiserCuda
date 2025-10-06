#include "image.cuh"
#include "filter.cuh"

void test_image(){
    Image<uchar3> image_uchar3("render/cornell/1/render.png");
    image_uchar3.save("build/output/uchar3.png");    

    Image<float3> image_float3("render/cornell/1/render.png");
    image_float3.save("build/output/float3.png");
}

void test_cudacpy(){
    Image<float3> image("render/cornell/1/render.png");
    CudaVector<float3> vec(image.vecBuffer);
    vec.copy(image.vecBuffer);
    image.save("build/output/cpy.png");
}


void test_filter_cpu(){
    Image<float3> render_img("render/cornell/1/render.png");
    Image<float3> albedo_img("render/cornell/1/albedo.png");
    Image<float3> normal_img("render/cornell/1/normal.png");
    
    int2 shape = render_img.shape;

    Image<float3> denoised_img(shape);

    waveletfilterCPU(
        {shape, render_img.vecBuffer.data(), normal_img.vecBuffer.data(), albedo_img.vecBuffer.data(), denoised_img.vecBuffer.data()},
        {5, 1.0f, 2.0f, 0.5f, 0.3f}
    );    
    //cudaMemcpy(denoised_img.vecBuffer.data(), denoised.data(), sizeof(float3) * totalSize(shape), cudaMemcpyDeviceToHost);
 
    denoised_img.save("build/output/denoised_cpu.png");
}

void test_filter_gpu(){
    Image<float3> render_img("render/cornell/1/render.png");
    Image<float3> albedo_img("render/cornell/1/albedo.png");
    Image<float3> normal_img("render/cornell/1/normal.png");
    
    int2 shape = render_img.shape;

    Image<float3> denoised_img(shape);

    CudaVector<float3> render(render_img.vecBuffer);
    CudaVector<float3> albedo(albedo_img.vecBuffer);
    CudaVector<float3> normal(normal_img.vecBuffer);
    CudaVector<float3> denoised(totalSize(shape));

    waveletfilterGPU(
        {shape, render.data(), normal.data(), albedo.data(), denoised.data()},
        {5, 1.0f, 2.0f, 0.5f, 0.3f}
    );  
    //cudaMemcpy(denoised_img.vecBuffer.data(), denoised.data(), sizeof(float3) * totalSize(shape), cudaMemcpyDeviceToHost);
    denoised.copyTo(denoised_img.vecBuffer);
 
    denoised_img.save("build/output/denoised_gpu.png");
}


void test(){
    test_image();
    test_cudacpy();
    //test_filter_cpu();
    test_filter_gpu();
}