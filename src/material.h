#pragma once

#include "vec.h"

#define PI_OVER_TWO 1.5707963267948966192313216916397514420985

struct Medium
{
  Vec3 absCoefficient;
  float scatterCoefficient;
  int dummy[3];

  Medium(Vec3 absColor, float absDist, float scatterDist) : absCoefficient(log(absColor) / (-absDist)), scatterCoefficient(1.0f / scatterDist) {}

  Medium(float absColor, float absDist, float scatterDist) : absCoefficient(log(Vec3(absColor, absColor, absColor)) / (-absDist)), scatterCoefficient(1.0f / scatterDist) {}

  Medium(float absCoefficient, float scatterCoefficient) : absCoefficient(Vec3(absCoefficient, absCoefficient, absCoefficient)), scatterCoefficient(scatterCoefficient) {}

  Medium(Vec3 absCoefficient, float scatterCoefficient) : absCoefficient(absCoefficient), scatterCoefficient(scatterCoefficient) {}

  Medium() : absCoefficient(Vec3()), scatterCoefficient(1e20f) {}
};

struct Material
{
  Vec3 kd;
  Vec3 ke;
  float roughness;
  float ior;
  int type;
  int medIdx;
  int kdtex;
  int ketex;
  int d_tex;
  float d;

  Material(Vec3 kd_, Vec3 ke_, float rough_, float ior_, int type_, int kdtex_, int ketex_, int d_tex_, int medIdx_ = -1) : kd(kd_), ke(ke_), roughness(rough_), ior(ior_), type(type_), kdtex(kdtex_), ketex(ketex_), d_tex(d_tex_), medIdx(medIdx_), d(1.0) {}

  Material(Vec3 kd_, float rough_, float ior_, int type_, int kdtex_, int ketex_, int d_tex_, int medIdx_ = -1) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(rough_), ior(ior_), type(type_), kdtex(kdtex_), ketex(ketex_), d_tex(d_tex_), medIdx(medIdx_), d(1.0) {}

  Material(Vec3 kd_, float rough_, float ior_, int type_, int medIdx_ = -1) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(rough_), ior(ior_), type(type_), kdtex(-1), ketex(-1), d_tex(-1), medIdx(medIdx_), d(1.0) {}

  Material(Vec3 kd_, float rough_, int type_) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(rough_), ior(1.0f), type(type_), kdtex(-1), ketex(-1), d_tex(-1), medIdx(-1), d(1.0) {}

  Material(Vec3 kd_, Vec3 ke_, float rough_, int type_) : kd(kd_), ke(ke_), roughness(rough_), ior(1.0f), type(type_), kdtex(-1), ketex(-1), d_tex(-1), medIdx(-1), d(1.0) {}

  Material(Vec3 kd_, Vec3 ke_) : kd(kd_), ke(ke_), roughness(0.0f), ior(1.0f), type(0), kdtex(-1), ketex(-1), d_tex(-1), medIdx(-1), d(1.0) {}

  Material(Vec3 kd_) : kd(kd_), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(0.0f), ior(1.0f), type(0), kdtex(-1), ketex(-1), d_tex(-1), medIdx(-1), d(1.0) {}

  Material() : kd(Vec3(0.9f, 0.9f, 0.9f)), ke(Vec3(0.0f, 0.0f, 0.0f)), roughness(0.0f), ior(1.0f), type(0), kdtex(-1), ketex(-1), d_tex(-1), medIdx(-1), d(1.0) {}
};
