#include "filter.cuh"

#include "image.h"
#include "utils.h"

#include <math.h>
#include <regex>
#include <iostream>

__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};



/*CUDA_FUNC float normalLenght(float3 v){
    return length(v/255.0);
}

CUDA_FUNC float waveletWeight(int2 pos, int2 n, int2 d, const Pixel* in, const GBuffer& frame, const FilterParams params){
    float3 dCol = make_float3(in[flattenIndex(pos, frame.shape)] - in[flattenIndex(n, frame.shape)]);
    float wCol = normalLenght(dCol)/params.sigmaColor;

    float3 dAlbedo = make_float3(frame.albedo[flattenIndex(pos, frame.shape)] - frame.albedo[flattenIndex(n, frame.shape)]);
    float wAlbedo = normalLenght(dAlbedo)/params.sigmaAlbedo;

    float dNormal = min(0.0, dot(frame.normal[flattenIndex(pos, frame.shape)], frame.normal[flattenIndex(n, frame.shape)]));
    float wNormal = dNormal/params.sigmaNormal;

    float wSpace = length(make_float2(d))/params.sigmaSpace;
    float wWavelet = waveletSpline[abs(d.x)]*waveletSpline[(abs(d.y))];

    return wWavelet*exp(-wCol-wSpace-wAlbedo-wNormal);
}*/