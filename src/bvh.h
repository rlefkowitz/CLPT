#pragma once

#include "primitives.h"
#include <algorithm>
#include "spice.h"

using namespace std;

const int resMax = 64;

int bvhaxis;

struct AABB
{
  union
  {
    Vec3 box[2];
    float _v[8];
  };

  AABB() : box{Vec3(1e20f, 1e20f, 1e20f), Vec3(-1e20f, -1e20f, -1e20f)}
  {
  }

  AABB(Vec3 b[]) : box{b[0], b[1]} {}

  AABB(const AABB &b) : box{b.box[0], b.box[1]} {}

  AABB(Vec3 a, Vec3 b) : box{a, b} {}

  inline Vec3 mp() { return ((box[0] + box[1]) * 0.5f); }

  inline Vec3 span() { return (box[1] - box[0]); }

  inline float sa()
  {
    Vec3 sp = span();
    return sp.x * sp.y + sp.y * sp.z + sp.x * sp.z;
  }

  inline void grow(Vec3 pt)
  {
    box[0] = min3(box[0], pt);
    box[1] = max3(box[1], pt);
  }

  inline void grow(AABB capt)
  {
    box[0] = min3(box[0], capt.box[0]);
    box[1] = max3(box[1], capt.box[1]);
  }
};

AABB boxFromTriangle(Triangle triangle)
{
  return AABB(Vec3(
                  min(triangle.v0.x, min(triangle.v1.x, triangle.v2.x)),
                  min(triangle.v0.y, min(triangle.v1.y, triangle.v2.y)),
                  min(triangle.v0.z, min(triangle.v1.z, triangle.v2.z))),
              Vec3(
                  max(triangle.v0.x, max(triangle.v1.x, triangle.v2.x)),
                  max(triangle.v0.y, max(triangle.v1.y, triangle.v2.y)),
                  max(triangle.v0.z, max(triangle.v1.z, triangle.v2.z))));
}

struct TriangleBox
{
  Triangle tri;
  AABB box;
  Vec3 mp;
  TriangleData data;

  TriangleBox(const TriangleBox &tb) : tri(tb.tri), box(tb.box), mp(tb.mp), data(tb.data) {}

  TriangleBox(Triangle t, AABB b, TriangleData d) : tri(t), box(b), mp(b.mp()), data(d) {}

  bool operator>(const TriangleBox &x) const
  {
    return (tri > x.tri);
  }

  bool operator<(const TriangleBox x) const
  {
    return mp[bvhaxis] < x.mp[bvhaxis];
  }
};

struct Split
{
  int axis;
  float pos;
  float cost;
  int binId;
};

struct BVHNode
{
  AABB box;
  int parent, child1, child2, isLeaf;
  int dummy[4];

  BVHNode() : box(), parent(), child1(), child2(), isLeaf()
  {
  }

  BVHNode(const BVHNode &other) noexcept
      : box(), parent(), child1(), child2(), isLeaf()
  {
    box = other.box;
    parent = other.parent;
    child1 = other.child1;
    child2 = other.child2;
    isLeaf = other.isLeaf;
  }
};

struct Bin
{
  AABB bb;
  int count;

  Bin() : bb(), count(0) {}
  Bin(AABB bb, int count) : bb(bb), count(count) {}
  Bin(TriangleBox tri) : bb(tri.box), count(1) {}
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

void addPrimitive(Bin &b, TriangleBox t)
{
  b.count++;
  b.bb.grow(t.box);
}

const float traversalStepCost = 0.5f;
const float primitiveIsectCost = 1.0f;

void makeLeaf(vector<BVHNode> &bvh, vector<TriangleBox> &tris, vector<Triangle> &flatTriangles, vector<TriangleData> &flatTriangleData, BVHNode node, int parentidx, int whichchild)
{
  node.isLeaf = 1;

  sort(tris.begin(), tris.end(), greater<TriangleBox>());

  vector<Triangle> ftris;
  ftris.clear();

  vector<TriangleData> ftriData;
  ftriData.clear();

  for (TriangleBox tri : tris)
  {
    ftris.push_back(tri.tri);
    ftriData.push_back(tri.data);
  }

  node.child1 = flatTriangles.size();
  flatTriangles.insert(flatTriangles.end(), ftris.begin(), ftris.end());
  flatTriangleData.insert(flatTriangleData.end(), ftriData.begin(), ftriData.end());
  node.child2 = flatTriangles.size();
  if (parentidx != -1)
  {
    if (whichchild == 1)
      bvh[parentidx].child1 = bvh.size();
    else
      bvh[parentidx].child2 = bvh.size();
  }
  bvh.push_back(node);
}

void build(vector<BVHNode> &bvh, vector<TriangleBox> &tris, vector<Triangle> &flatTriangles, vector<TriangleData> &flatTriangleData, AABB aabb, int parentidx, int whichchild, int depth)
{
  if (depth < 4)
    cout << depth << endl;

  BVHNode node;
  node.parent = parentidx;
  node.isLeaf = 0;

  // Copy parent first
  node.box = aabb;

  int triCount = tris.size();

  float leafCost = primitiveIsectCost * triCount;

  // End is reached
  if (triCount < 4 || depth > 30)
  {
    makeLeaf(bvh, tris, flatTriangles, flatTriangleData, node, parentidx, whichchild);
    return;
  }

  float parentSA = node.box.sa();

  Split bs = {-1, 0.0f, (float)triCount, -1};

  AABB centroidbox;

  for (TriangleBox tri : tris)
    centroidbox.grow(tri.mp);

  Vec3 axiscomp = centroidbox.span();

  // float max = axiscomp[0];
  // int axis_opt = 0;
  // for (int i = 1; i < 3; i++)
  //   if (axiscomp[i] > max)
  //   {
  //     max = axiscomp[i];
  //     axis_opt = i;
  //   }

  int res = resMax;

  bool altRes = false;
  // triCount <= res;

  vector<vector<float>> splits(3);

  Bin bins[res];

  AABB leftBoxes[res - 1];
  AABB rightBoxes[res - 1];
  int leftSums[res - 1];
  int rightSums[res - 1];

  AABB minLeftBox, minRightBox;
  float minCost = INFINITY;
  int minCostSplitBucket = -1;
  int minCostSplitAxis;

  // for (int axis = axis_opt; axis <= axis_opt; axis++)
  for (int axis = 0; axis < 3; axis++)
  {

    if (altRes)
    {
      bvhaxis = axis;
      sort(tris.begin(), tris.end());
      for (TriangleBox tri : tris)
        if (splits[axis].size() < 1 || splits[axis].back() != tri.mp[axis])
          splits[axis].push_back(tri.mp[axis]);
      // splits[axis].pop_back();
      res = splits[axis].size();
    }

    if (abs(axiscomp[axis]) < 1e-20f)
      continue;
    // printf("Binning for axis %d...\n", axis);

    // printf("Reinitializing bins...\n");
    for (int i = 0; i < res; i++)
      bins[i] = Bin();
    // printf("Reinitialized bins\n");

    if (altRes)
    {
      // printf("Doing process for res = %d\n", res);
      int j = 0;
      for (int b = 0; b < res; b++)
      {
        int spl = splits[axis][b];
        while (tris[j].mp[axis] <= spl)
        {
          addPrimitive(bins[b], tris[j]);
          j++;
        }
      }
      // for (; j < triCount; j++)
      // {
      //   addPrimitive(bins[res - 1], tris[j]);
      // }
    }
    else
    {
      for (TriangleBox tri : tris)
      {
        // printf("Calculating b with %f, %f, and %f...\n", tri.mp[axis], centroidbox.box[0][axis], axiscomp[axis]);
        int b = res * ((tri.mp[axis] - centroidbox.box[0][axis]) / axiscomp[axis]);
        // printf("Calculated b\n");
        if (b == res)
          b = res - 1;
        // printf("Adding primitive to %d...\n", b);
        addPrimitive(bins[b], tri);
        // printf("Added primitive to %d\n", b);
      }
    }

    // printf("Binned for axis %d\n", axis);

    // printf("Fitting boxes for axis %d...\n", axis);

    leftBoxes[0] = AABB(bins[0].bb.box);
    leftSums[0] = bins[0].count;

    rightBoxes[res - 2] = AABB(bins[res - 1].bb.box);
    rightSums[res - 2] = bins[res - 1].count;

    for (int i = 1; i < res - 1; i++)
    {
      leftBoxes[i] = AABB(leftBoxes[i - 1].box);
      leftBoxes[i].grow(bins[i].bb.box);
      leftSums[i] = leftSums[i - 1] + bins[i].count;

      rightBoxes[res - i - 2] = AABB(rightBoxes[res - i - 1].box);
      rightBoxes[res - i - 2].grow(bins[res - i - 1].bb);
      rightSums[res - i - 2] = rightSums[res - i - 1] + bins[res - i - 1].count;
    }

    // printf("Fitted boxes for axis %d\n", axis);

    // printf("Calculating costs for axis %d...\n", axis);

    float cost[res - 1];
    for (int i = 0; i < res - 1; ++i)
    {
      if (leftSums[i] >= 1 && rightSums[i] >= 1)
        cost[i] = traversalStepCost + primitiveIsectCost * (leftSums[i] * leftBoxes[i].sa() + rightSums[i] * rightBoxes[i].sa()) / parentSA;
      else
        cost[i] = INFINITY;
    }

    for (int i = 0; i < res - 1; ++i)
      if (cost[i] < minCost)
      {
        minCost = cost[i];
        minCostSplitBucket = i;
        minCostSplitAxis = axis;
        minLeftBox = leftBoxes[i];
        minRightBox = rightBoxes[i];
      }

    // printf("Calculated costs for axis %d\n", axis);
  }

  if (leafCost <= 255 && minCost >= leafCost)
  {
    makeLeaf(bvh, tris, flatTriangles, flatTriangleData, node, parentidx, whichchild);
    return;
  }

  int axis = minCostSplitAxis;
  node.isLeaf = 0;

  int thisidx = bvh.size();
  bvh.push_back(node);
  if (parentidx != -1)
  {
    if (whichchild == 1)
      bvh[parentidx].child1 = thisidx;
    else
      bvh[parentidx].child2 = thisidx;
  }

  AABB leftAABB = minLeftBox, rightAABB = minRightBox;

  vector<TriangleBox>::iterator mid;
  if (altRes)
  {
    mid = partition(tris.begin(), tris.end(),
                    [=](const TriangleBox &tri)
                    {
                      return tri.mp[axis] < splits[axis][minCostSplitBucket - 1];
                    });
  }
  else
  {
    mid = partition(tris.begin(), tris.end(),
                    [=](const TriangleBox &tri)
                    {
                      int b = res * ((tri.mp[axis] - centroidbox.box[0][axis]) / axiscomp[axis]);
                      if (b == res)
                        b = res - 1;
                      return b <= minCostSplitBucket;
                    });
  }

  vector<TriangleBox> leftTris(tris.begin(), mid);
  vector<TriangleBox> rightTris(mid, tris.end());

  build(bvh, leftTris, flatTriangles, flatTriangleData, leftAABB, thisidx, 1, depth + 1);

  build(bvh, rightTris, flatTriangles, flatTriangleData, rightAABB, thisidx, 2, depth + 1);
}

vector<BVHNode> build(vector<Triangle> &triangles, vector<TriangleData> &triangleData)
{
  vector<BVHNode> bvh;
  bvh.clear();
  vector<Triangle> flatTriangles;
  flatTriangles.clear();
  vector<TriangleData> flatTriangleData;
  flatTriangleData.clear();

  vector<TriangleBox> tris;
  tris.clear();
  for (int i = 0; i < triangles.size(); i++)
  {
    tris.push_back(TriangleBox(triangles[i], boxFromTriangle(triangles[i]), triangleData[i]));
  }

  AABB aabb = tris[0].box;
  for (int i = 1; i < tris.size(); i++)
  {
    aabb.grow(tris[i].box);
  }
  build(bvh, tris, flatTriangles, flatTriangleData, aabb, -1, 1, 0);
  tris.clear();
  triangles.clear();
  triangles.insert(triangles.end(), flatTriangles.begin(), flatTriangles.end());
  triangleData.clear();
  triangleData.insert(triangleData.end(), flatTriangleData.begin(), flatTriangleData.end());

  for (Triangle &t : triangles)
  {
    makeRenderReady(t);
  }
  // for (BVHNode node : bvh)
  // {

  // }

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
