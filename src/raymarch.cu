#include "raymarch.cuh"

#include "matrix.cuh"
#include "third_party/helper_math.h"


const int MAX_STEPS = 100;
const float MAX_DIST = 1e4;
const float SURF_DIST = 1e-6;

float2 operator/(const int2& a, const int2& b) {
    return make_float2((float) a.x/b.x, (float) a.y/b.y);
}

float3 viewportToWorld(int2 pos, int2 shape, Camera camera) {
    float3 worldUp = {0,1,0};
    float3 right = normalize(cross(camera.forward, worldUp));
    float3 up    = normalize(cross(right, camera.forward));

    float u = (2.0f * pos.x / (float)shape.x - 1.0f);
    float v = (2.0f * pos.y / (float)shape.y - 1.0f);

    float3 planeCenter = camera.forward * camera.dist;
    float3 posInPlane = (u * camera.plane.x * 0.5f) * right + (v * camera.plane.y * 0.5f) * up;

    return camera.pos + planeCenter + posInPlane;
}


float sdfSphere(float3 pos, float r){
    return length(pos)-r;
}

float sdfPlane(float3 pos){
    return -pos.y;
}


RenderData raymarchPoint(float3 pos, Scene scene){
    RenderData data;
    data.depth = MAX_DIST;
    
    for(int i = 0; i < scene.size; i++){
        Solid solid = scene.solids[i];
        float currDist;

        switch (solid.type){
        case Sphere:
            currDist = sdfSphere(pos, solid.scale.x);
            break;
        case Box:
           //currDist = sdfBox(pos);
        default:
            break;
        }

        if(currDist < data.depth){
            data.depth = currDist;
            data.id = i;
            if(i == 1)
                data.col = make_float3(1,0,0);
            if(i == 0)
                data.col = make_float3(0,0,1);
        }
    }
    return data;
}

float3 raymarchNormal(float3 p, Scene scene) {
	float d = raymarchPoint(p, scene).depth;
    float e = .01;
    
    float3 n = d - make_float3(
        raymarchPoint(p-make_float3(e,0,0), scene).depth,
        raymarchPoint(p-make_float3(0,e,0), scene).depth,
        raymarchPoint(p-make_float3(0,0,e), scene).depth
    );
    
    return normalize(n);
}

RenderData raymarchRay(Ray ray, Scene scene){
    RenderData data;
    float distTotal = 0;
    float distStep = 0;
    float3 posCurr;

    for(int i = 0; i < MAX_STEPS; i++) {
        distTotal += distStep;
        if(distTotal > MAX_DIST){
            data.id = -1;
            data.col = {0,0,0};
            data.normal = {0,0,0};
            break;
        }

    	posCurr = ray.origin + ray.direction * distTotal;
        data = raymarchPoint(posCurr, scene);
        distStep = data.depth;
        
        if(distStep < SURF_DIST) break;
    }

    if(data.id != -1)
        data.normal = raymarchNormal(posCurr, scene);

    return data;
}

void raymarchSceneCPU(Camera camera, Scene scene, Framebuffer framebuffer){
    int2 pixelPos = {0,0};
    for(pixelPos.x = 0; pixelPos.x < framebuffer.shape.x; pixelPos.x++){
        for(pixelPos.y = 0; pixelPos.y < framebuffer.shape.y; pixelPos.y++){
            raymarchScenePixel(pixelPos, camera, scene, framebuffer);
        }
    }
}


void raymarchSceneGPU(Camera camera, Scene scene, Framebuffer framebuffer){
    dim3 blockSize(16, 16);
    dim3 gridSize((framebuffer.shape.x + 15) / 16, (framebuffer.shape.y + 15) / 16);

    raymarchSceneKernel<<<gridSize,blockSize>>>(camera, scene, framebuffer);
    cudaDeviceSynchronize();
}


KERNEL void raymarchSceneKernel(Camera camera, Scene scene, Framebuffer framebuffer){
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if(pos.x >= framebuffer.shape.x || pos.y >= framebuffer.shape.y)
        return;
    
    raymarchScenePixel(pos, camera, scene, framebuffer);
}

KFUNC void raymarchScenePixel(int2 pixelPos, Camera camera, Scene scene, Framebuffer framebuffer){
    float3 worldPos = viewportToWorld(pixelPos, framebuffer.shape, camera);
    Ray ray = {camera.pos, normalize(worldPos - camera.pos)};
    RenderData data = raymarchRay(ray, scene);
    framebuffer.render[index(pixelPos, framebuffer.shape)] = data.col;
    framebuffer.normal[index(pixelPos, framebuffer.shape)] = data.normal;

    framebuffer.render[index(pixelPos, framebuffer.shape)] = {255, 0, 0};
}
