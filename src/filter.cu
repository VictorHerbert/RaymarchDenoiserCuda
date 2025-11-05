#include "filter.cuh"

#include "image.cuh"
#include "utils.cuh"

#include <math.h>
#include <regex>
#include <iostream>

__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

const int2 MAX_BLOCK_SIZE = {26,26};
const int MAX_KERNEL_RADIUS = 2;
const int SHARED_MEM_OCUPANCY = sizeof(uchar3)*(MAX_BLOCK_SIZE.y + 2 * MAX_KERNEL_RADIUS)*(MAX_BLOCK_SIZE.y + 2 * MAX_KERNEL_RADIUS);

static_assert(SHARED_MEM_OCUPANCY <= (48 * 1024), "Shared memory occupancy exceeds threshold");

KERNEL void filterKernel(Framebuffer frame, FilterParams params){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if(pos.x >= frame.shape.x || pos.y >= frame.shape.y)
        return;

    __shared__ uchar3 tile[MAX_BLOCK_SIZE.y + 2 * MAX_KERNEL_RADIUS][MAX_BLOCK_SIZE.x + 2 * MAX_KERNEL_RADIUS];
    int2 blockPos = {threadIdx.x, threadIdx.y};

    // Global index
    int idx = row * width + col;

    // Load central pixel
    if (col < width && row < height)
        tile[ty + KERNEL_RADIUS][tx + KERNEL_RADIUS] = in[idx];

    // Load halo pixels (borders)
    // Left and right
    if (tx < KERNEL_RADIUS) {
        int leftCol = max(col - KERNEL_RADIUS, 0);
        int rightCol = min(col + blockDim.x, width - 1);
        if (row < height) {
            tile[ty + KERNEL_RADIUS][tx] = in[row * width + leftCol];
            tile[ty + KERNEL_RADIUS][tx + blockDim.x + KERNEL_RADIUS] = in[row * width + rightCol];
        }
    }

    // Top and bottom
    if (ty < KERNEL_RADIUS) {
        int topRow = max(row - KERNEL_RADIUS, 0);
        int bottomRow = min(row + blockDim.y, height - 1);
        if (col < width) {
            tile[ty][tx + KERNEL_RADIUS] = in[topRow * width + col];
            tile[ty + blockDim.y + KERNEL_RADIUS][tx + KERNEL_RADIUS] = in[bottomRow * width + col];
        }
    }

    for(int i = 0; i < params.depth; i++){
        //printf("%i/%i\n", i, params.depth);
        
        params.step = 1<<i;
        filterPixel(
            pos,
            (i == 0) ? frame.render : frame.buffer[i%2],
            (i == (params.depth - 1)) ? frame.denoised : frame.buffer[(i+1)%2],
            frame, params);

        __syncthreads();
    }
}

CUDA_FUNC void filterPixel(int2 pos, const Pixel* in, Pixel* out, const Framebuffer frame, const FilterParams params){
    
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

            float w;
            switch (params.type){
            case FilterParams::AVERAGE:
                w = averageWeight(pos, n, d, in, frame, params);
                break;
            case FilterParams::WAVELET:
                w = waveletWeight(pos, n, d, in, frame, params);        
            default:
                break;
            }
            
            acum += w*in[index(n, frame.shape)];
            norm += w;
        }
    }
    acum /= norm;
    out[index(pos, frame.shape)] = make_uchar3(acum);
}

CUDA_FUNC float averageWeight(int2 pos, int2 n, int2 d, const Pixel* in, const Framebuffer& frame, const FilterParams params){
    return 1.0f; // Equal weight
}

CUDA_FUNC float normalLenght(float3 v){
    return length(v/255.0);
}

CUDA_FUNC float waveletWeight(int2 pos, int2 n, int2 d, const Pixel* in, const Framebuffer& frame, const FilterParams params){
    float3 dCol = make_float3(in[index(pos, frame.shape)] - in[index(n, frame.shape)]);
    float wCol = normalLenght(dCol)/params.sigmaColor;

    float3 dAlbedo = make_float3(frame.albedo[index(pos, frame.shape)] - frame.albedo[index(n, frame.shape)]);
    float wAlbedo = normalLenght(dAlbedo)/params.sigmaAlbedo;

    float dNormal = min(0.0, dot(frame.normal[index(pos, frame.shape)], frame.normal[index(n, frame.shape)]));
    float wNormal = dNormal/params.sigmaNormal;

    float wSpace = length(make_float2(d))/params.sigmaSpace;
    float wWavelet = waveletSpline[abs(d.x)]*waveletSpline[(abs(d.y))];

    return wWavelet*exp(-wCol-wSpace-wAlbedo-wNormal);
}


