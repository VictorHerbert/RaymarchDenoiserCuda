#ifndef GBUFFER_H
#define GBUFFER_H

template<typename T>
struct GBuffer{
    int2 shape;

    T* render;
    T* denoised;
    T* normal;
    T* albedo;
    T* buffer[2];
};

#endif