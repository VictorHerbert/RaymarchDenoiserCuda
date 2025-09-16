#ifndef RAYMARCH_H
#define RAYMARCH_H

#include "utils.h"

const int PARAM_LIM = 6;

struct Camera {
    float3 pos;
    float3 forward;
    float2 plane;
    float dist;
};

enum SolidType{Sphere, Box};

struct Solid{
    SolidType type;
    float3 pos;
    float3 scale;
    float3 col;

};

struct Scene{
    int size;
    Solid* solids;
};

struct Framebuffer{
    int2 shape;
    float3* render;
    float3* normal;
    float3* albedo;
};

struct Ray{
    float3 origin;
    float3 direction;
};

struct RenderData{
    int id;
    float depth;
    float3 col;
    float3 normal;
};

float sdfSphere(float3 pos, float r);
float sdfBox(float3 pos);

void raymarchSceneCPU(Camera camera, Scene scene, Framebuffer frame);
void raymarchSceneGPU(Camera camera, Scene scene, Framebuffer frame);
KERNEL void raymarchSceneKernel(Camera camera, Scene scene, Framebuffer frame);
KFUNC void raymarchScenePixel(int2 pixelPos, Camera camera, Scene scene, Framebuffer framebuffer);


#endif