#pragma once

#include "primitives.h"
#include <algorithm>

using namespace std;

// void print(std::vector<float> const &input) {
//     for (int i = 0; i < input.size(); i++) {
//         cout << (float) input.at(i) << ' ';
//     }
//     cout << endl;
// }

struct TriNumPair
{
  Triangle tri;
  float cst;

  TriNumPair(Triangle t, const float &c) : tri(t), cst(c) {}

  bool operator<(const TriNumPair &tnp) const
  {
    return (cst < tnp.cst);
  }

  bool operator>(const TriNumPair &tnp) const
  {
    return (cst > tnp.cst);
  }
};

//void print(std::vector<int> const &input)
//{
//    for (int i = 0; i < input.size(); i++) {
//        cout << input.at(i) << ' ';
//    }
//    cout << endl;
//}

const int res = 16;

struct Split
{
  int axis;
  float pos;
  float cost;
  int binId;
};

struct BVHNode
{
  Vec3 box[2];
  int parent, child1, child2, isLeaf;

  inline Vec3 span() const { return (box[1] - box[0]); }

  inline float sa() const
  {
    Vec3 sp = span();
    return abs(2.0f * (sp.x * sp.y + sp.y * sp.z + sp.x * sp.z));
  }
};

struct Bin
{
  BVHNode bb;
  vector<Triangle> triangles;
};

bool setIfBetterSplit(Split &past, Split &curr)
{
  if (curr.cost < past.cost && curr.cost > 0)
  {
    past.axis = curr.axis;
    past.pos = curr.pos;
    past.cost = curr.cost;
    past.binId = curr.binId;
    return true;
  }
  return false;
}

Bin initBin()
{
  Bin newBin;
  newBin.triangles.clear();
  newBin.bb.box[0] = Vec3(1e20f, 1e20f, 1e20f);
  newBin.bb.box[1] = Vec3(-1e20f, -1e20f, -1e20f);
  return newBin;
}

void grow(BVHNode &node, Vec3 pt)
{
  node.box[0] = min3(node.box[0], pt);
  node.box[1] = max3(node.box[1], pt);
}

void grow(BVHNode &node, BVHNode toCapture)
{
  node.box[0] = min3(node.box[0], toCapture.box[0]);
  node.box[1] = max3(node.box[1], toCapture.box[1]);
}

BVHNode makeBoxFromTriangle(Triangle triangle)
{
  BVHNode aabb;
  aabb.box[0] = Vec3(
      min(triangle.v0.x, triangle.v1.x, triangle.v2.x),
      min(triangle.v0.y, triangle.v1.y, triangle.v2.y),
      min(triangle.v0.z, triangle.v1.z, triangle.v2.z));
  aabb.box[1] = Vec3(
      max(triangle.v0.x, triangle.v1.x, triangle.v2.x),
      max(triangle.v0.y, triangle.v1.y, triangle.v2.y),
      max(triangle.v0.z, triangle.v1.z, triangle.v2.z));
  return aabb;
}

Vec3 triBoxMP(Triangle triangle)
{
  BVHNode aabb;
  aabb.box[0] = Vec3(
      min(triangle.v0.x, triangle.v1.x, triangle.v2.x),
      min(triangle.v0.y, triangle.v1.y, triangle.v2.y),
      min(triangle.v0.z, triangle.v1.z, triangle.v2.z));
  aabb.box[1] = Vec3(
      max(triangle.v0.x, triangle.v1.x, triangle.v2.x),
      max(triangle.v0.y, triangle.v1.y, triangle.v2.y),
      max(triangle.v0.z, triangle.v1.z, triangle.v2.z));
  return (aabb.span() / 2.0f + aabb.box[0]);
}

void addPrimitive(Bin &b, Triangle t)
{
  b.triangles.push_back(t);
  grow(b.bb, makeBoxFromTriangle(t));
}

void swap(float *xp, float *yp)
{
  float temp = *xp;
  *xp = *yp;
  *yp = temp;
}

void swap(Triangle *xp, Triangle *yp)
{
  Triangle temp = *xp;
  *xp = *yp;
  *yp = temp;
}

const float traversalStepCost = 1.0f;
const float primitiveIsectCost = 2.0f;

void build(vector<BVHNode> &bvh, vector<Triangle> &triangles, vector<Triangle> &flatTriangles, BVHNode aabb, int parentidx, int whichchild, int depth)
{
  if (depth < 4)
    cout << depth << endl;

  BVHNode node;
  node.parent = parentidx;
  node.isLeaf = 0;

  vector<Triangle> tris;
  tris.clear();
  for (Triangle tri : triangles)
  {
    tris.push_back(tri);
  }
  node.box[0] = Vec3(aabb.box[0]);
  node.box[1] = Vec3(aabb.box[1]);
  //    cout << tris.size() << endl;

  if (tris.size() < 4 || depth > 30)
  {
    //        cout << tris.size() << endl;
    node.isLeaf = 1;

    std::sort(tris.begin(), tris.end(), greater<Triangle>());

    node.child1 = flatTriangles.size();
    flatTriangles.insert(flatTriangles.end(), tris.begin(), tris.end());
    node.child2 = flatTriangles.size();
    if (parentidx != -1)
    {
      if (whichchild == 1)
      {
        bvh[parentidx].child1 = bvh.size();
      }
      else
      {
        bvh[parentidx].child2 = bvh.size();
      }
    }
    bvh.push_back(node);
    return;
  }

  float parentSA = node.sa();

  Split bs = {-1, 0.0f, primitiveIsectCost * (float)tris.size(), -1};
  //    printf("SurfArea of current node: %f\n", node.sa());
  //    printf("Triangle count in current node: %d\n", tris.size());
  //    printf("Cost of current node: %f\n", bs.cost);

  BVHNode centroidbox;
  centroidbox.box[0] = triBoxMP(tris[0]);
  centroidbox.box[1] = Vec3(centroidbox.box[0]);

  for (int i = 1; i < tris.size(); i++)
  {
    grow(centroidbox, triBoxMP(tris[i]));
  }

  Vec3 axiscomp = centroidbox.span();

  // float max = axiscomp[0];
  // int axis = 0;
  // for(int i = 1; i < 3; i++) {
  //     if(axiscomp[i] > max) {
  //         max = axiscomp[i];
  //         axis = i;
  //     }
  // }

  vector<vector<Triangle>> right_tris(3);
  vector<vector<Triangle>> left_tris(3);
  vector<vector<Bin>> bins(3);
  vector<vector<BVHNode>> leftBoxes(3);
  vector<vector<BVHNode>> rightBoxes(3);
  vector<vector<int>> leftSums(3);
  vector<vector<int>> rightSums(3);

  for (int axis = 0; axis < 3; axis++)
  {

    bins[axis] = vector<Bin>(res);

    float testinterval = axiscomp[axis] / ((float)res);
    float testpos = centroidbox.box[0][axis] + testinterval;

    right_tris[axis].clear();
    left_tris[axis].clear();

    bvhaxis = axis;

    std::sort(tris.begin(), tris.end());

    right_tris[axis].insert(right_tris[axis].end(), tris.begin(), tris.end());

    int rps = right_tris[axis].size();

    for (int i = 0; i < res; i++, testpos += testinterval)
    {

      bins[axis][i] = initBin();

      while (rps > 0 && right_tris[axis][0].boxMP(axis) <= testpos)
      {
        addPrimitive(bins[axis][i], right_tris[axis][0]);
        right_tris[axis].erase(right_tris[axis].begin());
        rps--;
      }
    }

    while (rps > 0)
    {
      addPrimitive(bins[axis][res - 1], right_tris[axis][0]);
      right_tris[axis].erase(right_tris[axis].begin());
      rps--;
    }

    leftBoxes[axis] = vector<BVHNode>(res - 1);
    rightBoxes[axis] = vector<BVHNode>(res - 1);
    leftSums[axis] = vector<int>(res - 1);
    rightSums[axis] = vector<int>(res - 1);
    for (int i = 0; i < res - 1; i++)
    {
      BVHNode lnode;
      leftBoxes[axis][i] = lnode;
      BVHNode rnode;
      rightBoxes[axis][i] = rnode;
    }

    leftBoxes[axis][0].box[0] = Vec3(bins[axis][0].bb.box[0]);
    leftBoxes[axis][0].box[1] = Vec3(bins[axis][0].bb.box[1]);
    leftSums[axis][0] = bins[axis][0].triangles.size();

    rightBoxes[axis][res - 2].box[0] = Vec3(bins[axis][res - 1].bb.box[0]);
    rightBoxes[axis][res - 2].box[1] = Vec3(bins[axis][res - 1].bb.box[1]);
    rightSums[axis][res - 2] = bins[axis][res - 1].triangles.size();

    for (int i = 1; i < res - 1; i++)
    {
      leftBoxes[axis][i].box[0] = Vec3(leftBoxes[axis][i - 1].box[0]);
      leftBoxes[axis][i].box[1] = Vec3(leftBoxes[axis][i - 1].box[1]);
      grow(leftBoxes[axis][i], bins[axis][i].bb);
      leftSums[axis][i] = leftSums[axis][i - 1] + bins[axis][i].triangles.size();

      rightBoxes[axis][res - i - 2].box[0] = Vec3(rightBoxes[axis][res - i - 1].box[0]);
      rightBoxes[axis][res - i - 2].box[1] = Vec3(rightBoxes[axis][res - i - 1].box[1]);
      grow(rightBoxes[axis][res - i - 2], bins[axis][res - i - 1].bb);
      rightSums[axis][res - i - 2] = rightSums[axis][res - i - 1] + bins[axis][res - i - 1].triangles.size();
    }

    testpos = node.box[0][axis] + testinterval;

    for (int i = 0; i < res - 1; i++, testpos += testinterval)
    {
      if (leftSums[axis][i] >= 1 && rightSums[axis][i] >= 1)
      {
        float nscost = traversalStepCost + primitiveIsectCost / parentSA * (leftBoxes[axis][i].sa() * leftSums[axis][i] + rightBoxes[axis][i].sa() * rightSums[axis][i]);
        Split curr = {axis, testpos, nscost, i};
        setIfBetterSplit(bs, curr);
      }
    }
  }

  int axis = bs.axis;

  for (int i = 0; i < 3; i++)
  {
    if (i != axis)
    {
      right_tris[i].clear();
      left_tris[i].clear();
      bins[i].clear();
      leftBoxes[i].clear();
      rightBoxes[i].clear();
    }
    leftSums[i].clear();
    rightSums[i].clear();
  }
  leftSums.clear();
  rightSums.clear();

  if (bs.axis == -1)
  {

    node.isLeaf = 1;

    std::sort(tris.begin(), tris.end(), greater<Triangle>());

    node.child1 = flatTriangles.size();
    flatTriangles.insert(flatTriangles.end(), tris.begin(), tris.end());
    node.child2 = flatTriangles.size();
    if (parentidx != -1)
    {
      if (whichchild == 1)
      {
        bvh[parentidx].child1 = bvh.size();
      }
      else
      {
        bvh[parentidx].child2 = bvh.size();
      }
    }
    bvh.push_back(node);
    return;
  }

  node.isLeaf = bs.axis << 1;

  int thisidx = bvh.size();
  bvh.push_back(node);
  if (parentidx != -1)
  {
    if (whichchild == 1)
    {
      bvh[parentidx].child1 = thisidx;
    }
    else
    {
      bvh[parentidx].child2 = thisidx;
    }
  }

  //    printf("Split. Cost: %f\n", bs.cost);

  left_tris[axis].clear();
  right_tris[axis].clear();

  int b = 0;
  for (; b < bs.binId + 1; b++)
    left_tris[axis].insert(left_tris[axis].end(), bins[axis][b].triangles.begin(), bins[axis][b].triangles.end());
  for (; b < res; b++)
    right_tris[axis].insert(right_tris[axis].end(), bins[axis][b].triangles.begin(), bins[axis][b].triangles.end());

  bins[axis].clear();
  bins.clear();

  BVHNode tempL;
  tempL.box[0] = Vec3(leftBoxes[axis][bs.binId].box[0]);
  tempL.box[1] = Vec3(leftBoxes[axis][bs.binId].box[1]);

  leftBoxes[axis].clear();
  leftBoxes.clear();

  build(bvh, left_tris[axis], flatTriangles, tempL, thisidx, 1, depth + 1);
  left_tris[axis].clear();
  left_tris.clear();

  BVHNode tempR;
  tempR.box[0] = Vec3(rightBoxes[axis][bs.binId].box[0]);
  tempR.box[1] = Vec3(rightBoxes[axis][bs.binId].box[1]);

  rightBoxes[axis].clear();
  rightBoxes.clear();

  build(bvh, right_tris[axis], flatTriangles, tempR, thisidx, 2, depth + 1);
  right_tris[axis].clear();
  right_tris.clear();
}

vector<BVHNode> build(vector<Triangle> &triangles)
{
  vector<BVHNode> bvh;
  bvh.clear();
  vector<Triangle> flatTriangles;
  flatTriangles.clear();

  BVHNode aabb = makeBoxFromTriangle(triangles[0]);
  aabb.isLeaf = 0;
  aabb.parent = -1;
  for (int i = 1; i < triangles.size(); i++)
  {
    grow(aabb, makeBoxFromTriangle(triangles[i]));
  }
  build(bvh, triangles, flatTriangles, aabb, -1, 1, 0);
  triangles.clear();
  triangles.insert(triangles.end(), flatTriangles.begin(), flatTriangles.end());

  for (Triangle &t : triangles)
  {
    makeRenderReady(t);
  }

  //    vector<int> parentlist;
  //    parentlist.clear();
  //    vector<int> child1list;
  //    child1list.clear();
  //    vector<int> child2list;
  //    child2list.clear();
  //    for(BVHNode node : bvh) {
  //        parentlist.push_back(node.parent);
  //        child1list.push_back(node.child1);
  //        child2list.push_back(node.child2);
  //    }
  //    print(parentlist);
  //    print(child1list);
  //    print(child2list);

  return bvh;
}
