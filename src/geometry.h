#pragma once

#include "linear_algebra.h"
#include "material.h"

struct Sphere
{
    Vec3 pos;
    float radius;
    int mtlidx;
    int dummy[2];
    
    Sphere(float r, Vec3 p, int mi) : radius(r), pos(p), mtlidx(mi) {}
    
    Sphere() : radius(1.0f), pos(Vec3(0.0f, 0.0f, 0.0f)), mtlidx(0) {}
    
};

struct Triangle
{
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    int mtlidx;
    int dummy[3];
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
