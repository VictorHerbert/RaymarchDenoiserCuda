#ifndef RAYMARCH_H
#define RAYMARCH_H

#include "utils.cuh"
#include "image.cuh"

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
    size_t size;
    Solid* solids;
};

struct Ray{
    float3 origin;
    float3 direction;
};

struct RenderData{
    size_t id;
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

KFUNC float3 viewportToWorld(int2 pos, int2 shape, Camera camera);
KFUNC RenderData raymarchPoint(float3 pos, Scene scene);
KFUNC float3 raymarchNormal(float3 p, Scene scene);
KFUNC RenderData raymarchRay(Ray ray, Scene scene);
KFUNC float3 ambientLight(float3 normal, Scene scene);

#endif