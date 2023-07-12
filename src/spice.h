#pragma once

#include "vec.h";

struct TextureData
{
  int w;
  int h;
  int s;
  int dummy;
};

struct TriangleData
{
  Vec3 t0;
  Vec3 t1;
  Vec3 t2;
  Vec3 n0;
  Vec3 n1;
  Vec3 n2;
  Vec3 uv;
  Vec3 vv;
};