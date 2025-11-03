#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "cuda_runtime.h"

#include "utils.cuh"
#include "vector.cuh"

typedef uchar3 Pixel;

struct Framebuffer{
    int2 shape;
    Pixel* render;
    Pixel* normal;
    Pixel* albedo;
    Pixel* denoised;

    Pixel* buffer[2];
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

struct CPUFramebuffer : Framebuffer{
    Image render, albedo, normal;
};

struct CudaFramebuffer : Framebuffer {
    CudaVector<Pixel> renderVec, albedoVec, normalVec, denoisedVec;
    CudaVector<Pixel> bufferVec;
    CPUVector<Pixel> denoisedVecCpu;

    CudaFramebuffer();
    CudaFramebuffer (int2 shape);

    void allocate(int2 shape);

    void openImages(std::string filepath, cudaStream_t stream = 0);
};

#endif