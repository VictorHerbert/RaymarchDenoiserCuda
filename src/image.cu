#include "image.cuh"

#include <string>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include "third_party/helper_math.h"

#include "utils.cuh"
#include <stdexcept>
#include <regex>


Image::Image(int2 shape){
    this->shape = shape;
    vecBuffer.resize(totalSize(shape));
    data = vecBuffer.data();
}

Image::Image(uchar3* data, int2 shape){
    this->shape = shape;
    vecBuffer.resize(totalSize(shape));
    data = vecBuffer.data();
}

Image::Image(std::string filename){
    int dummy;
    //auto start = std::chrono::high_resolution_clock::now();
    uchar3* buffer = (uchar3*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &dummy, 3);
    //auto mid = std::chrono::high_resolution_clock::now();

    if(buffer == nullptr)
        throw std::runtime_error("Failed to load image" + filename + "': " + stbi_failure_reason());

    vecBuffer.resize(totalSize(shape));
    memcpy(vecBuffer.data(), buffer, totalSize(shape) * sizeof(uchar3));
    
    free(buffer);
    data = vecBuffer.data();
    //auto end = std::chrono::high_resolution_clock::now();

    //double total = std::chrono::duration<double, std::milli>(end - start).count();
    //double read = std::chrono::duration<double, std::milli>(mid - start).count();
    //double copy = std::chrono::duration<double, std::milli>(end - mid).count();

    //printf("Image %s: %f %f %f ms\n", filename.c_str(), read, copy, total);
}

void Image::save(std::string filename){
    if(!stbi_write_png(filename.c_str(), shape.x, shape.y, 3, vecBuffer.data(), shape.x * 3)){
        throw std::runtime_error("Failed to save image " + filename + "': " + stbi_failure_reason());
    }
}

void Image::save(std::string filename, uchar3* data, int2 shape){
    if(!stbi_write_png(filename.c_str(), shape.x, shape.y, 3, data, shape.x * 3)){
        throw std::runtime_error("Failed to save image " + filename + "': " + stbi_failure_reason());
    }
}

CudaFramebuffer::~CudaFramebuffer(){
    cudaFreeHost(denoisedCPU);
}

CudaFramebuffer::CudaFramebuffer(int2 shape){
    this->shape = shape;

    int size = totalSize(shape);
    
    renderVec.resize(size);
    albedoVec.resize(size);
    normalVec.resize(size);
    denoisedVec.resize(size);
    bufferVec.resize(2*size);

    render = renderVec.data();
    albedo = albedoVec.data();
    normal = normalVec.data();
    denoised = denoisedVec.data();
    buffer[0] = bufferVec.data();
    buffer[1] = bufferVec.data() + size;
}

void CudaFramebuffer::allocate(int2 shape){
    this->shape = shape;

    int size = totalSize(shape);
    
    renderVec.resize(size);
    albedoVec.resize(size);
    normalVec.resize(size);
    denoisedVec.resize(size);
    bufferVec.resize(2*size);
    cudaMallocHost(&denoisedCPU, size * sizeof(Pixel));

    render = renderVec.data();
    albedo = albedoVec.data();
    normal = normalVec.data();
    denoised = denoisedVec.data();
    buffer[0] = bufferVec.data();
    buffer[1] = bufferVec.data() + size;
}

void saveImage(std::string filepath, Pixel* data, int2 shape){
    if(!stbi_write_png(filepath.c_str(), shape.x, shape.y, 3, data, shape.x * 3)){
        throw std::runtime_error("Failed to save image " + filepath + "': " + stbi_failure_reason());
    }
}

void CudaFramebuffer::openImages(std::string filepath, cudaStream_t stream){
    int byteCount = sizeof(Pixel) * totalSize(shape);
    std::regex pattern(R"(\$type\$)");

    Image render_img(std::regex_replace(filepath, pattern, "render"));
    cudaMemcpyAsync(render, render_img.data, byteCount, cudaMemcpyHostToDevice, stream);
    Image albedo_img(std::regex_replace(filepath, pattern, "albedo"));
    cudaMemcpyAsync(albedo, albedo_img.data, byteCount, cudaMemcpyHostToDevice, stream);
    Image normal_img(std::regex_replace(filepath, pattern, "normal"));
    cudaMemcpyAsync(normal, normal_img.data, byteCount, cudaMemcpyHostToDevice, stream);
}


/*CPUFramebuffer::CPUFramebuffer(std::string filepath){
    fromImages(filepath);
}

void CPUFramebuffer::fromImages(std::string filepath){
    std::regex pattern(R"(\$type\$)");
    Image render_img(std::regex_replace(filepath, pattern, "render"));
    shape = render_img.shape;
    render = render_img;
    albedo = Image(std::regex_replace(filepath, pattern, "albedo"));
    normal = Image(std::regex_replace(filepath, pattern, "normal"));
}

void CudaFramebuffer::fromImages(std::string filepath){
    fromCPUFrame(CPUFramebuffer(filepath));
}

void CudaFramebuffer::fromCPUFrame(CPUFramebuffer cpuFrame){
    //renderVec.from(cpuFrame.)
}*/