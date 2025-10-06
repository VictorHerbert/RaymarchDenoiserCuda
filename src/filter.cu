#include "filter.cuh"

#include <math.h>
#include "third_party/helper_math.h"
#include "iostream"

#include "image.cuh"
#include "raymarch.cuh"

void waveletfilterCPU(Framebuffer frame, DenoiseParams params){
    static CPUVector<float3> bufferVec;
    if(2 * totalSize(frame.shape) > bufferVec.size())
        bufferVec.resize(2 * totalSize(frame.shape));

    float3* buffer[2] = {bufferVec.data(), bufferVec.data() + totalSize(frame.shape)};

    DenoiseParams pixelParams = params;

    for(int i = 0; i < params.depth; i++){
        int2 pos = {0,0};
        for(pos.x = 0; pos.x < frame.shape.x; pos.x++){
            pixelParams.step = 1<<i;
            for(pos.y = 0; pos.y < frame.shape.y; pos.y++){
                waveletfilterPixel(
                    pos,
                    i == 0 ? frame.render : buffer[i%2],
                    i == (params.depth - 1) ? frame.denoised : buffer[(i+1)%2],
                    frame, pixelParams
                );
            }
        }
    }
}

void waveletfilterGPU(Framebuffer frame, DenoiseParams params){  
    dim3 blockSize(16, 16);
    dim3 gridSize((frame.shape.x + 15) / 16, (frame.shape.y + 15) / 16);
    DenoiseParams pixelParams = params;

    static CudaVector<float3> bufferVec;
    if(2 * totalSize(frame.shape) > bufferVec.size())
        bufferVec.resize(2 * totalSize(frame.shape));

    float3* buffer[2] = {bufferVec.data(), bufferVec.data() + totalSize(frame.shape)};

    for(int i = 0; i < params.depth; i++){
        pixelParams.step = 1<<i;
        waveletKernel<<<gridSize,blockSize>>>(
            (i == 0) ? frame.render : buffer[i%2],
            (i == (params.depth - 1)) ? frame.denoised : buffer[(i+1)%2],
            frame, pixelParams);

        cudaDeviceSynchronize();
    }
}

__global__ void waveletKernel(float3* in, float3* out, Framebuffer frame, DenoiseParams params){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if(pos.x >= frame.shape.x || pos.y >= frame.shape.y)
        return;

    waveletfilterPixel(pos, in, out, frame, params);
}

void waveletfilterPixel(int2 pos, float3* in, float3* out, Framebuffer frame, DenoiseParams params){
    const float h[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

    float3 acum = {0, 0, 0};
    float norm = 0;

    int2 d;
    for(d.x = -2; d.x <= 2; d.x++){
        for(d.y = -2; d.y <= 2; d.y++){
            int2 n = {
                pos.x + d.x * params.step,
                pos.y + d.y * params.step
            };

            if(inRange(n, frame.shape)){
                float3 dCol = in[index(pos, frame.shape)] - in[index(n, frame.shape)];
                float wCol = length(dCol)/params.sigmaColor;

                float3 dAlbedo = frame.albedo[index(pos, frame.shape)] - frame.albedo[index(n, frame.shape)];
                float wAlbedo = length(dAlbedo)/params.sigmaAlbedo;

                float dNormal = min(0.0, dot(frame.normal[index(pos, frame.shape)], frame.normal[index(n, frame.shape)]));
                float wNormal = dNormal/(params.sigmaNormal*params.step*params.step);

                float wSpace = length(make_float2(d))/params.sigmaSpace;
                float wWavelet = h[abs(d.x)]*h[(abs(d.y))];

                wSpace = 0;
                //wNormal = 0;
                //wAlbedo = 0;

                float w = wWavelet*exp(-wCol-wSpace-wAlbedo-wNormal);

                acum += w*in[index(n, frame.shape)];
                norm += w;
            }
        }
    }
    acum /= norm;
    out[index(pos, frame.shape)] = acum;
}