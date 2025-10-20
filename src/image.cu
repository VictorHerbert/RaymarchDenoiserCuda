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

Image::Image(std::string filename){
    int dummy;
    uchar3* buffer = (uchar3*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &dummy, 3);

    if(buffer == nullptr)
        throw std::runtime_error("Failed to load image" + filename + "': " + stbi_failure_reason());

    vecBuffer.resize(totalSize(shape));
    memcpy(vecBuffer.data(), buffer, totalSize(shape) * sizeof(uchar3));
    
    free(buffer);
}

void Image::save(std::string filename){
    if(!stbi_write_png(filename.c_str(), shape.x, shape.y, 3, vecBuffer.data(), shape.x * 3)){
        throw std::runtime_error("Failed to save image" + filename + "': " + stbi_failure_reason());
    }
}