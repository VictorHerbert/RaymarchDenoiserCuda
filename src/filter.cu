#include "filter.cuh"

#include "image.cuh"
#include "utils.cuh"

#include <math.h>
#include <regex>
#include <iostream>

__constant__ float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};

KERNEL void filterKernel(Framebuffer frame, const FilterParams params){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };    
    
    //cacheTile(albedoTile, frame.albedo, frame.shape, params.radius);
    //cacheTile(normalTile, frame.normal, frame.shape, params.radius);

    int2 blockShape = { blockDim.x, blockDim.y };
    int2 blockPos   = { threadIdx.x, threadIdx.y };
    int2 halo       = { params.radius, params.radius };

    int2 tileShape = blockShape + 2*halo;
    int tileSize = totalSize(tileShape);

    extern __shared__ uchar3 tile[];
    uchar3 *renderTile = tile;
    uchar3 *bufferTile = renderTile + tileSize;
    
    for(int level = 0; level < params.depth; level++){     
        uchar3* in = (level == 0) ? frame.render : frame.buffer[level%2];
        uchar3* out = (level == (params.depth - 1)) ? frame.denoised : frame.buffer[(level+1)%2];

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
                if(params.cacheInput)
                    acum += w*renderTile[flattenIndex(nTilePos, tileShape)];
                else
                    acum += w*in[flattenIndex(nPos, frame.shape)];

                norm += w;
            }
        }
        acum /= norm;
        out[flattenIndex(pos, frame.shape)] = make_uchar3(acum);

        __syncthreads();
    }
}

CUDA_FUNC void cacheTile(uchar3* tile, uchar3* in, int2 shape, int radius){
    int2 gridPos    = { blockIdx.x, blockIdx.y };
    int2 blockShape = { blockDim.x, blockDim.y };
    int2 blockPos   = { threadIdx.x, threadIdx.y };
    int2 halo       = { radius, radius };

    int2 tileSize = { blockShape.x + 2 * halo.x, blockShape.y + 2 * halo.y };
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


CUDA_FUNC float averageWeight(int2 pos, int2 n, int2 d, const Pixel* in, const Framebuffer& frame, const FilterParams params){
    return 1.0f; // Equal weight
}

CUDA_FUNC float normalLenght(float3 v){
    return length(v/255.0);
}

CUDA_FUNC float waveletWeight(int2 pos, int2 n, int2 d, const Pixel* in, const Framebuffer& frame, const FilterParams params){
    float3 dCol = make_float3(in[flattenIndex(pos, frame.shape)] - in[flattenIndex(n, frame.shape)]);
    float wCol = normalLenght(dCol)/params.sigmaColor;

    float3 dAlbedo = make_float3(frame.albedo[flattenIndex(pos, frame.shape)] - frame.albedo[flattenIndex(n, frame.shape)]);
    float wAlbedo = normalLenght(dAlbedo)/params.sigmaAlbedo;

    float dNormal = min(0.0, dot(frame.normal[flattenIndex(pos, frame.shape)], frame.normal[flattenIndex(n, frame.shape)]));
    float wNormal = dNormal/params.sigmaNormal;

    float wSpace = length(make_float2(d))/params.sigmaSpace;
    float wWavelet = waveletSpline[abs(d.x)]*waveletSpline[(abs(d.y))];

    return wWavelet*exp(-wCol-wSpace-wAlbedo-wNormal);
}


