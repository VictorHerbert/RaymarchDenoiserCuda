#include "filter.cuh"

#include <math.h>
#include "iostream"
#include "image.cuh"
#include "utils.cuh"

void waveletfilterCPU(Framebuffer frame, DenoiseParams params){
    static CPUVector<Pixel> bufferVec;
    if(2 * totalSize(frame.shape) > bufferVec.size())
        bufferVec.resize(2 * totalSize(frame.shape));

    Pixel* buffer[2] = {bufferVec.data(), bufferVec.data() + totalSize(frame.shape)};

    const float scaleSigma = 3*255*2;
    DenoiseParams pixelParams{
        params.depth,
        params.sigmaSpace * scaleSigma,
        params.sigmaColor * scaleSigma,
        params.sigmaAlbedo * scaleSigma,
        params.sigmaNormal * scaleSigma
    };

    if(params.depth == 0)
        memcpy(frame.denoised, frame.render, sizeof(uchar3) * totalSize(frame.shape));

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

float square(float f){
    return f*f;
}

void waveletfilterGPU(Framebuffer frame, DenoiseParams params){
    dim3 blockSize(16, 16);
    dim3 gridSize((frame.shape.x + 15) / 16, (frame.shape.y + 15) / 16);
    float scaleSigma = 3*255*2;
    scaleSigma = 1;
    DenoiseParams pixelParams{
        params.depth,
        params.sigmaSpace * scaleSigma,
        params.sigmaColor * scaleSigma,
        params.sigmaAlbedo * scaleSigma,
        params.sigmaNormal * scaleSigma
    };

    static CudaVector<Pixel> bufferVec;
    if(2 * totalSize(frame.shape) > bufferVec.size())
        bufferVec.resize(2 * totalSize(frame.shape));

    Pixel* buffer[2] = {bufferVec.data(), bufferVec.data() + totalSize(frame.shape)};

    for(int i = 0; i < params.depth; i++){
        pixelParams.step = 1<<i;
        waveletKernel<<<gridSize,blockSize>>>(
            (i == 0) ? frame.render : buffer[i%2],
            (i == (params.depth - 1)) ? frame.denoised : buffer[(i+1)%2],
            frame, pixelParams);

        cudaDeviceSynchronize();
    }
}

KERNEL void waveletKernel(Pixel* in, Pixel* out, Framebuffer frame, DenoiseParams params){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if(pos.x >= frame.shape.x || pos.y >= frame.shape.y)
        return;

    waveletfilterPixel(pos, in, out, frame, params);
}


float normalLenght(float3 v){
    return length(v/255.0);
}

KFUNC void waveletfilterPixel(int2 pos, Pixel* in, Pixel* out, Framebuffer frame, DenoiseParams params){
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

            if(!inRange(n, frame.shape))
                continue;

            float3 dCol = make_float3(in[index(pos, frame.shape)] - in[index(n, frame.shape)]);
            float wCol = normalLenght(dCol)/params.sigmaColor;

            float3 dAlbedo = make_float3(frame.albedo[index(pos, frame.shape)] - frame.albedo[index(n, frame.shape)]);
            float wAlbedo = normalLenght(dAlbedo)/params.sigmaAlbedo;

            float dNormal = min(0.0, dot(frame.normal[index(pos, frame.shape)], frame.normal[index(n, frame.shape)]));
            float wNormal = dNormal/params.sigmaNormal;

            float wSpace = length(make_float2(d))/params.sigmaSpace;
            float wWavelet = h[abs(d.x)]*h[(abs(d.y))];

            float w = wWavelet*exp(-wCol-wSpace-wAlbedo-wNormal);
            w = 1;

            acum += w*in[index(n, frame.shape)];
            norm += w;
        }
    }
    acum /= norm;
    //clamp(acum, {0,0,0}, {255, 255, 255});
    out[index(pos, frame.shape)] = make_uchar3(acum*255);
}