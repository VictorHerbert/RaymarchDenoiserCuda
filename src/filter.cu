#include "filter.cuh"

#include <math.h>
#include "third_party/helper_math.h"
#include "iostream"



void waveletfilterCPU(
    int2 shape,  int depth,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
){

    std::vector<float3> arena(2*totalSize(shape));
    memcpy(arena.data(), in, sizeof(float3) * totalSize(shape));

    float3* buffer[2] = {arena.data(), arena.data() + totalSize(shape)};

    for(int i = 0; i < depth; i++){
        int2 pos = {0,0};
        for(pos.x = 0; pos.x < shape.x; pos.x++){
            for(pos.y = 0; pos.y < shape.y; pos.y++){
                waveletfilterPixel(
                    pos, shape, 1<<i,
                    buffer[i%2], buffer[(i+1)%2], variance, albedo, normal,
                    sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal 
                );
            }
        }
    }

    memcpy(out, buffer[depth%2], sizeof(float3) * totalSize(shape));
}

void waveletfilterGPU(
    int2 shape, int depth,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
){
    dim3 blockSize(16, 16);
    dim3 gridSize((shape.x + 15) / 16, (shape.y + 15) / 16);


    cudaDeviceSynchronize();

    /*float3* buffer[2] = {b1.data, b2.data};
    for(int i = 0; i < depth; i++){
        waveletKernel<<<gridSize,blockSize>>>(
            shape, 1<<(i+1),
            buffer[i%2], buffer[(i+1)%2], variance, albedo, normal,
            sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal 
        );
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(out, buffer[depth%2], sizeof(float3)*totalSize(shape), cudaMemcpyDeviceToHost); */
}

__global__ void waveletKernel(    
    int2 shape, int step,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
){
    
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if(pos.x >= shape.x || pos.y >= shape.y)
        return;
    
    waveletfilterPixel(
        pos, shape, step,
        in, out, variance, albedo, normal,
        sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal 
    );
}

void waveletfilterPixel(
    int2 pos, int2 shape, int step,
    float3* in, float3* out, float* variance, float3* albedo, float3* normal, 
    float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal
){
    const float h[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

    float3 acum = {0, 0, 0};
    float norm = 0;

    int2 d;
    for(d.x = -2; d.x <= 2; d.x++){
        for(d.y = -2; d.y <= 2; d.y++){
            int2 n = {
                pos.x + d.x * step,
                pos.y + d.y * step
            };

            if(inRange(n, shape)){
                float3 dCol = in[index(pos, shape)] - in[index(n, shape)];
                float wCol = length(dCol)/sigmaColor;

                float3 dAlbedo = albedo[index(pos, shape)] - albedo[index(n, shape)];
                float wAlbedo = length(dAlbedo)/sigmaAlbedo;

                float dNormal = min(0.0, dot(normal[index(pos, shape)], normal[index(n, shape)]));
                float wNormal = dNormal/(sigmaNormal*step*step);
                
                float wSpace = length(make_float2(d))/sigmaSpace;
                float wWavelet = h[abs(d.x)]*h[(abs(d.y))];

                wSpace = 0;
                //wNormal = 0;
                //wAlbedo = 0;

                float w = wWavelet*exp(-wCol-wSpace-wAlbedo-wNormal);
          
                acum += w*in[index(n, shape)];
                norm += w;
            }
        }
    }
    acum /= norm;
    out[index(pos, shape)] = acum;
}