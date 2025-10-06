#include "image.cuh"

#include <string>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include "third_party/helper_math.h"

#include "utils.cuh"
#include <stdexcept>

template<>
Image<uchar3>::Image(std::string filename){
    int dummy;
    uchar3* buffer = (uchar3*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &dummy, 3);

    if(buffer == nullptr)
        throw std::runtime_error("Failed to load image" + filename + "': " + stbi_failure_reason());

    vecBuffer.resize(totalSize(shape));
    memcpy(vecBuffer.data(), buffer, totalSize(shape) * sizeof(uchar3));
    
    free(buffer);
}

template<>
Image<float3>::Image(std::string filename){
    int dummy;
    uchar3* buffer = (uchar3*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &dummy, 3);

    if(buffer == nullptr)
        throw std::runtime_error("Failed to load image" + filename + "': " + stbi_failure_reason());

    vecBuffer.resize(totalSize(shape));
    for(int i = 0; i < vecBuffer.size(); i++){
        vecBuffer[i] = make_float3(buffer[i].x, buffer[i].y, buffer[i].z);
        vecBuffer[i] /= 255;
    }

    free(buffer);
}

template<>
void Image<uchar3>::save(std::string filename){
    if(!stbi_write_png(filename.c_str(), shape.x, shape.y, 3, vecBuffer.data(), shape.x * 3)){
        throw std::runtime_error("Failed to save image" + filename + "': " + stbi_failure_reason());
    }
}

template<>
void Image<float3>::save(std::string filename){
    std::vector<uchar3> buffer(totalSize(shape));
    for(int i = 0; i < buffer.size(); i++){
        buffer[i] = make_uchar3(vecBuffer[i].x*255, vecBuffer[i].y*255, vecBuffer[i].z*255);
    }

    if(!stbi_write_png(filename.c_str(), shape.x, shape.y, 3, buffer.data(), shape.x * 3)){
        throw std::runtime_error("Failed to save image" + filename + "': " + stbi_failure_reason());
    }
}