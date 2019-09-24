#pragma once

#include "linear_algebra.h"

#define PI_OVER_TWO 1.5707963267948966192313216916397514420985

struct Material
{
    Vec3 kd;
    Vec3 ke;
    float roughness;
    float ior;
    int tex0;
    int tex1;
    int type;
    int dummy[3];
    
    Material(Vec3 kd_, Vec3 ke_, float rough_, float ior_, int tex0_, int tex1_, int type_) : kd(kd_), ke(ke_), roughness(rough_), ior(ior_), tex0(tex0_), tex1(tex1_), type(type_) {}
    
    Material(Vec3 kd_, float rough_, float ior_, int tex0_, int tex1_, int type_) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(rough_), ior(ior_), tex0(tex0_), tex1(tex1_), type(type_) {}
    
    Material(Vec3 kd_, float rough_, float ior_, int type_) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(rough_), ior(ior_), tex0(-1), tex1(-1), type(type_) {}
    
    Material(Vec3 kd_, float rough_, int type_) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(rough_), ior(1.0f), tex0(-1), tex1(-1), type(type_) {}
    
    Material(Vec3 kd_, Vec3 ke_) : kd(kd_), ke(ke_), roughness(0.0f), ior(1.0f), tex0(-1), tex1(-1), type(0) {}
    
    Material(Vec3 kd_) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(0.0f), ior(1.0f), tex0(-1), tex1(-1), type(0) {}
    
    Material() : kd(Vec3(0.9f, 0.9f, 0.9f)), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(0.0f), ior(1.0f), tex0(-1), tex1(-1), type(0) {}
};
