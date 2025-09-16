#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "cuda_runtime.h"

#include "matrix.cuh"

struct Image{
    int3 shape;
    uchar* buffer;
    std::vector<uchar> vBuffer;
    bool stbi_allocated = false;
    
    Image(){}
    Image(float3* data, int2 shape);
    Image(std::string filename);
    ~Image();
    
    bool close();
    bool save(std::string filename);
};

std::vector<float3> fVecFromImage(const Image& img);

#endif