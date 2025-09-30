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

enum SolidType{Light, Sphere, Box};

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
    float3* denoised;
};

struct Ray{
    float3 origin;
    float3 direction;
};

struct RenderData{
    int id;
    float depth;
    float3 albedo;
    float3 col;
    float3 normal;
    float3 light;
};

//KFUNC float sdfSphere(float3 pos, float r);
//KFUNC float sdfBox(float3 pos, float3 dim);

void raymarchSceneCPU(Camera camera, Scene scene, Framebuffer frame);
void raymarchSceneGPU(Camera camera, Scene scene, Framebuffer frame);
KERNEL void raymarchSceneKernel(Camera camera, Scene scene, Framebuffer frame);
KFUNC void raymarchScenePixel(int2 pixelPos, Camera camera, Scene scene, Framebuffer framebuffer);

__forceinline__ KFUNC float3 viewportToWorld(int2 pos, int2 shape, Camera camera);
__forceinline__ KFUNC RenderData raymarchPoint(float3 pos, Scene scene);
__forceinline__ KFUNC float3 raymarchNormal(float3 p, Scene scene);
__forceinline__ KFUNC RenderData raymarchRay(Ray ray, Scene scene);
__forceinline__ KFUNC float3 ambientLight(float3 normal, Scene scene);

#endif