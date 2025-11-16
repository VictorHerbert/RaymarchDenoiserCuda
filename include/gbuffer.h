#ifndef GBUFFER_H
#define GBUFFER_H

#include "image.h"

template<typename T>
struct GBuffer{
    int2 shape;

    T* render;
    T* denoised;
    T* normal;
    T* albedo;
    T* buffer[2];
};

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