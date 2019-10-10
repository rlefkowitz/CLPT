#pragma once

#include "vec.h"
#include "material.h"

using namespace std;

struct Sphere
{
    Vec3 pos;
    float radius;
    int mtlidx;
    int dummy[2];
    
    Sphere(float r, Vec3 p, int mi) : radius(r), pos(p), mtlidx(mi) {}
    
    Sphere() : radius(1.0f), pos(Vec3(0.0f, 0.0f, 0.0f)), mtlidx(0) {}
    
};

int bvhaxis;

struct Triangle
{
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    int mtlidx;
    int dummy[3];
    
    float boxMP(int axis) const {
        float bMin = v0[axis];
        float bMax = bMin;
        bMin = min(bMin, v1[axis]);
        bMax = max(bMax, v1[axis]);
        return (max(bMax, v2[axis]) + min(bMin, v2[axis])) / 2.0f;
    }
    
    bool operator > (const Triangle& tri) const {
        Vec3 nuv00 = cross(v1 - v0, v2 - v0);
        float nuv0len0 = sqrtf(nuv00.lengthsq());
        Vec3 nuv01 = cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
        float nuv0len1 = sqrtf(nuv01.lengthsq());
        return (nuv0len0 > nuv0len1);
    }
    
    bool operator < (const Triangle& tri) const {
        return (boxMP(bvhaxis) < tri.boxMP(bvhaxis));
    }
    
};

struct Mesh
{
    Vec3 box[2];
    int tri0;
    int trin;
};

Triangle makeTriangle(Vec3 v0, Vec3 v1, Vec3 v2, int mtlidx) {
    Triangle t;
    t.mtlidx = mtlidx;
    t.v0 = v0;
    t.v1 = v1;
    t.v2 = v2;
    return t;
}
