#pragma once

#include <vector>
#include "bvh.h"
#include "camera.h"
#include "primitives.h"
#include "vec.h"
#include "spice.h"
#include "material.h"
#include "objloader.h"

using namespace std;

struct Scene
{
  vector<Sphere> spheres;
  vector<Triangle> triangles;
  vector<BVHNode> nodes;
  vector<Material> materials;
  vector<Medium> mediums;
  vector<TriangleData> triangleData;
  vector<string> textures;
  Vec3 cam_pos;
  Vec3 cam_fd;
  Vec3 cam_up;
  float cam_focal_distance;
  float cam_aperture_radius;
  string iblPath;
  Vec3 background_color;
  bool use_IbL;
  bool use_ground;
  bool use_DOF;
};
