#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "cuda_runtime.h"

#include "utils.cuh"
#include "vector.cuh"

typedef uchar3 Pixel;

template<typename T>
struct Framebuffer{
    int2 shape;
    T* render;
    T* denoised;
    T* normal;
    T* albedo;
    
    T* buffer[2];
};

struct Image{
    int2 shape;
    Pixel* data;
    std::vector<Pixel> vecBuffer;

    Image(){}
    Image(int2 shape);
    Image(uchar3* data, int2 shape);
    Image(std::string filename);
    
    void save(std::string filename);
    static void save(std::string filename, uchar3* data, int2 shape);
};

Pixel* openImage(std::string filepath);
void saveImage(std::string filepath, Pixel* data, int2 shape);

template<typename T>
struct CPUFramebuffer : Framebuffer<T>{
    Image render, albedo, normal;
};

template<typename T>
struct CudaFramebuffer : Framebuffer<T> {
    CudaVector<T> renderVec, albedoVec, normalVec, denoisedVec;
    CudaVector<T> bufferVec;
    //CPUVector<Pixel> denoisedVecCpu; // TODO remove
    T* denoisedCPU;

    CudaFramebuffer(){};
    ~CudaFramebuffer();
    CudaFramebuffer (int2 shape);

    void allocate(int2 shape);

    void openImages(std::string filepath, cudaStream_t stream = 0);
};

#endif