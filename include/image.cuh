#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "cuda_runtime.h"

#include "utils.cuh"
#include "vector.cuh"

struct Framebuffer{
    int2 shape;
    float3* render;
    float3* normal;
    float3* albedo;
    float3* denoised;
};

template <typename T>
struct Image{
    int2 shape;
    std::vector<T> vecBuffer;

    Image(int2 shape){
        this->shape = shape;
        vecBuffer.resize(totalSize(shape));
    }
    Image(std::string filename);
    
    void save(std::string filename);
};

#endif