#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "cuda_runtime.h"

#include "utils.cuh"
#include "vector.cuh"
#include "gbuffer.h"

typedef uchar3 Pixel;

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
struct CPUGBuffer : GBuffer<T>{
    Image render, albedo, normal;
};

template<typename T>
struct CudaGBuffer : GBuffer<T> {
    CudaVector<T> renderVec, albedoVec, normalVec, denoisedVec;
    CudaVector<T> bufferVec;
    //CPUVector<Pixel> denoisedVecCpu; // TODO remove
    T* denoisedCPU;

    CudaGBuffer(){};
    ~CudaGBuffer();
    CudaGBuffer (int2 shape);

    void allocate(int2 shape);

    void openImages(std::string filepath, cudaStream_t stream = 0);
};

#endif