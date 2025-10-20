#include "utils.cuh"


KFUNC int totalSize(int2 shape){
    return shape.x * shape.y;
}

KFUNC int totalSize(int3 shape){
    return shape.x * shape.y * shape.z;
}

KFUNC int inRange(int2 pos, int2 shape){
    return (pos.x >= 0) && (pos.x < shape.x) && (pos.y >= 0) && (pos.y < shape.y);
}

KFUNC int index(int2 p, int2 shape){
    return p.y * shape.x + p.x;
}

KFUNC uchar3 operator-(const uchar3 &a, const uchar3 &b) {
    return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

KFUNC float length(const uchar3 &v) {
    return sqrtf(float(v.x) * float(v.x) +
                 float(v.y) * float(v.y) +
                 float(v.z) * float(v.z));
}

KFUNC float dot(const uchar3 &a, const uchar3 &b) {
    return float(a.x) * float(b.x) +
           float(a.y) * float(b.y) +
           float(a.z) * float(b.z);
}

KFUNC float3 operator*(const float &f, const uchar3 &v) {
    return make_float3(f * v.x, f * v.y, f * v.z);
}

KFUNC float3 operator*(const uchar3 &v, const float &f) {
    return f * v;
}

KFUNC uchar3 make_uchar3(const float3 &v) {
    return uchar3{
        static_cast<unsigned char>(v.x),
        static_cast<unsigned char>(v.y),
        static_cast<unsigned char>(v.z)
    };
}