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
};

struct Image{
    int2 shape;
    std::vector<uchar3> vecBuffer;

    Image(int2 shape);
    Image(uchar3* data, int2 shape);
    Image(std::string filename);
    
    void save(std::string filename);
    static void save(std::string filename, uchar3* data, int2 shape);
};

#endif