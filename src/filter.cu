#include "filter.cuh"

#include "image.h"
#include "utils.h"

#include <math.h>
#include <regex>
#include <iostream>

__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};


KERNEL void filterKernelBaseline(GBuffer frame, const FilterParams params){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    int2 blockShape = { blockDim.x, blockDim.y };
    int2 blockPos   = { threadIdx.x, threadIdx.y };
    int2 halo       = { params.radius, params.radius };

    for(int level = 0; level < params.depth; level++){
        uchar4* in = (level == 0) ? frame.render : frame.buffer[level%2];
        uchar4* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];

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
                uchar4 mem = in[flattenIndex(nPos, frame.shape)];
                acum.x += w*mem.x;
                acum.y += w*mem.y;
                acum.z += w*mem.z;
                norm += w;                       
            }
        }
        acum /= norm;
        uchar4 mem;
        mem.x = (unsigned char) acum.x;
        mem.y = (unsigned char) acum.x;
        mem.z = (unsigned char) acum.x;
        out[flattenIndex(pos, frame.shape)] = mem;

        __syncthreads();
    }
}

CUDA_FUNC void cacheTile(uchar4* tile, uchar4* in, int2 shape, int radius){
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

            tile[idx] = in[frameIdx];
        }
    }

    __syncthreads();
}

KERNEL void filterKernelTiled(GBuffer frame, const FilterParams params){
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
        uchar4* in = (level == 0) ? frame.render : frame.buffer[level%2];
        uchar4* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];

        if(params.cacheInput)
            cacheTile(renderTile, in, frame.shape, params.radius);

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