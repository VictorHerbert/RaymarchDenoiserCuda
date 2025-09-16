#include "image.cuh"

#include <string>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include "third_party/helper_math.h"

#include <stdexcept>


Image::Image(float3* fmat, int2 shape){
    vBuffer.resize(3*totalSize(shape));
    this->shape = {shape.x, shape.y, 3};
    buffer = vBuffer.data();
    for(int i = 0; i < vBuffer.size(); i+=3){
        vBuffer[i] = static_cast<uchar>(fmat[i/3].x*255);
        vBuffer[i+1] = static_cast<uchar>(fmat[i/3].y*255);
        vBuffer[i+2] = static_cast<uchar>(fmat[i/3].z*255);
    }
    stbi_allocated = false;
}


Image::Image(std::string filename){    
    buffer = (uchar*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &(shape.z), 3);
    shape.z = 3;
    if(buffer == nullptr)
        throw std::runtime_error(
            std::string("Failed to load image '") + filename + "': " + stbi_failure_reason()
        );
    stbi_allocated = true;
}

Image::~Image(){
    if(stbi_allocated)
        close();
}

bool Image::close(){
    stbi_image_free(buffer);
    return true;
}

bool Image::save(std::string filename){
    return stbi_write_png(filename.c_str(), shape.x, shape.y, shape.z, buffer, shape.x * shape.z);
}

std::vector<float3> fVecFromImage(const Image& img){
    std::vector<float3> out(img.shape.x * img.shape.y);
    for(int i = 0; i < totalSize(img.shape); i+=3)
        out[i/3] = {
            static_cast<float>(img.buffer[i])/255,
            static_cast<float>(img.buffer[i+1])/255,
            static_cast<float>(img.buffer[i+2])/255
        };

    return out;
}