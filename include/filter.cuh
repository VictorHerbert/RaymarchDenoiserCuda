#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "utils.h"
#include "vector.h"
#include "image.h"

#include "extended_math.h"

struct FilterParams {
    enum FilterType {AVERAGE, GAUSSIAN, CROSS, WAVELET} type;
    int depth;
    int level;
    int radius;
    float sigmaSpace;
    float sigmaColor;
    float sigmaAlbedo;
    float sigmaNormal;

    bool cacheInput = true;
    bool cacheBuffer = true;
};


float3 snrCPU(Pixel* original, Pixel* noisy, int2 shape);
float3 snrGPU(Pixel* original, Pixel* noisy, int2 shape);

template<typename T>
CUDA_FUNC void cacheTile(uchar4* tile, T* in, int2 shape, int radius){
    int2 gridPos    = { blockIdx.x, blockIdx.y };
    int2 blockShape = { blockDim.x, blockDim.y };
    int2 blockPos   = { threadIdx.x, threadIdx.y };
    int2 halo       = { radius, radius };

    int2 tileSize = blockShape + 2 * halo;
    tileSize.x = (tileSize.x + blockShape.x-1) & ~(blockShape.x-1);
    int totalTileSize = tileSize.x * tileSize.y;

    int threadId = blockPos.y * blockShape.x + blockPos.x;

    for (int idx = threadId; idx < totalTileSize; idx += blockShape.x * blockShape.y) {
        int2 tilePos = { idx % tileSize.x, idx / tileSize.x };

        int2 framePos = gridPos * blockShape + tilePos - halo;

        if (inRange(framePos, shape)){
            int frameIdx = flattenIndex(framePos, shape);

            int address = (int)&in[frameIdx];
            int wordAddress = address/4;
            int bank = wordAddress%32;
            int lane = threadId%32;
            //printf("Tid %3d | Lane %2d | Addr 0x%x | WordAddr 0x%x | Bank %2d\n", threadId, lane, address, wordAddress, bank);
            //tile[idx] = make_uchar4(in[frameIdx].x,in[frameIdx].y,in[frameIdx].z, 0);
            tile[idx] = *(uchar4*)(&in[frameIdx]);

            //tile[idx].x = in[frameIdx].x;
            //tile[idx].y = in[frameIdx].y;
            //tile[idx].z = in[frameIdx].z;
        }
    }

    __syncthreads();
}

template <typename T>
KERNEL void filterKernel(GBuffer<T> frame, const FilterParams params){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 blockShape = { blockDim.x, blockDim.y };
    int2 blockPos   = { threadIdx.x, threadIdx.y };
    int2 halo       = { params.radius, params.radius };

    int2 tileShape = blockShape + 2*halo;
    int tileSize = totalSize(tileShape);

    extern __shared__ uchar4 tile[];
    uchar4 *renderTile = tile;

    for(int level = 0; level < params.depth; level++){
        T* in = (level == 0) ? frame.render : frame.buffer[level%2];
        T* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];

        if(params.cacheInput)
            cacheTile<T>(renderTile, in, frame.shape, params.radius);

        if(pos.x >= frame.shape.x || pos.y >= frame.shape.y)
            continue;

        int2 blockPos   = { threadIdx.x, threadIdx.y };

        float3 acum = {0, 0, 0};
        float norm = 0;
        int2 d;

        for(d.x = -params.radius; d.x <= params.radius; d.x++){
            for(d.y = -params.radius; d.y <= params.radius; d.y++){
                int2 nPos = pos + d;
                int2 nTilePos = blockPos + halo + d;

                if(!inRange(nPos, frame.shape))
                    continue;

                float w = 1;
                uchar4 memCache;
                if(params.cacheInput)
                    memCache = *(uchar4*)(&renderTile[flattenIndex(nTilePos, tileShape)]);
                else
                    memCache = *(uchar4*)(&in[flattenIndex(nPos, frame.shape)]);


                unsigned int threadId = blockPos.y * blockShape.x + blockPos.x;
                unsigned int address = (int)&renderTile[flattenIndex(nTilePos, tileShape)];
                unsigned int wordAddress = address/4;
                unsigned int bank = wordAddress%32;
                unsigned int lane = threadId%32;
                //printf("Tid %3d | Lane %2d | Addr 0x%x | WordAddr 0x%x | Bank %2d\n", threadId, lane, address, wordAddress, bank);

                acum.x += w*memCache.x;
                acum.y += w*memCache.y;
                acum.z += w*memCache.z;

                norm += w;
            }
        }
        acum /= norm;

        out[flattenIndex(pos, frame.shape)] = {
            static_cast<unsigned char>(acum.x),
            static_cast<unsigned char>(acum.y),
            static_cast<unsigned char>(acum.z),
        };
        __syncthreads();
    }
}

template <typename T>
KERNEL void filterKernelBaseline(GBuffer<T> frame, const FilterParams params){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 blockShape = { blockDim.x, blockDim.y };
    int2 blockPos   = { threadIdx.x, threadIdx.y };
    int2 halo       = { params.radius, params.radius };

    for(int level = 0; level < params.depth; level++){
        T* in = (level == 0) ? frame.render : frame.buffer[level%2];
        T* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];

        if(pos.x >= frame.shape.x || pos.y >= frame.shape.y)
            continue;

        float3 acum = {0, 0, 0};
        float norm = 0;

        int2 d;
        for(d.x = -params.radius; d.x <= params.radius; d.x++){
            for(d.y = -params.radius; d.y <= params.radius; d.y++){
                int2 nPos = pos + d;

                if(!inRange(nPos, frame.shape))
                    continue;

                float w = 1;
                T mem = in[flattenIndex(nPos, frame.shape)];
                acum.x += w*mem.x;
                acum.y += w*mem.y;
                acum.z += w*mem.z;
                norm += w;
            }
        }
        acum /= norm;
        T mem;
        mem.x = (unsigned char) acum.x;
        mem.y = (unsigned char) acum.x;
        mem.z = (unsigned char) acum.x;
        out[flattenIndex(pos, frame.shape)] = mem;

        __syncthreads();
    }
}

//CUDA_FUNC float waveletWeight(int2 pos, int2 n, int2 d, const Pixel* in, const GBuffer& frame, const FilterParams params);
//CUDA_FUNC float averageWeight(int2 pos, int2 n, int2 d, const Pixel* in, const GBuffer& frame, const FilterParams params);

void waveletFilterSequence(std::string inputPath, std::string outputPath);

#endif