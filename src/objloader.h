#pragma once

#include "primitives.h"
#include "parser.h"
#include "spice.h"
#include <algorithm>

using namespace std;
using namespace algorithm;

int loadMtl(vector<Material> &materials, vector<string> &textures, string model,
            string materialName, map<string, int> &texvars, int defaultValue)
{
  ifstream file("res/models/" + model + ".mtl");
  if (!file.is_open())
    return defaultValue;

  // printf("Loading material %s...\n", materialName.c_str());

  bool open = false;
  string objname;
  Vec3 kd = Vec3(0.0, 0.0, 0.0), ks = Vec3(0.0, 0.0, 0.0), ke = Vec3(0.0, 0.0, 0.0);
  string kdfile = "", kefile = "", dfile = "";
  bool pbrboost = false;
  bool noKdOnTex = false;
  float ns = 100.0f, ni = 1.0f, fr = 0.75f, d = 1.0f;
  int illum = 0;
  string line;
  bool foundMtl = false;
  while (getline(file, line))
  {
    string pref = firstToken(line);
    vector<string> currentLine;
    split(line, currentLine, " ");
    if (!foundMtl && pref == "newmtl" && currentLine[1].compare(materialName) == 0)
    {
      foundMtl = true;
    }
    else if (foundMtl && pref == "newmtl")
      break;
    else if (foundMtl)
    {
      if (pref == "map_Kd")
        kdfile = currentLine[1];
      else if (pref == "map_Ke")
        kefile = currentLine[1];
      else if (pref == "map_d")
        dfile = currentLine[1];
      else if (pref == "Kd")
      {
        kd.x = stof(currentLine[1]);
        kd.y = stof(currentLine[2]);
        kd.z = stof(currentLine[3]);
      }
      else if (pref == "Ks")
      {
        ks.x = stof(currentLine[1]);
        ks.y = stof(currentLine[2]);
        ks.z = stof(currentLine[3]);
      }
      else if (pref == "Ke")
      {
        ke.x = stof(currentLine[1]);
        ke.y = stof(currentLine[2]);
        ke.z = stof(currentLine[3]);
      }
      else if (pref == "illum")
        illum = stoi(currentLine[1]);
      else if (pref == "Ns")
        ns = stof(currentLine[1]);
      else if (pref == "Ni")
        ni = stof(currentLine[1]);
      else if (pref == "Fr")
        ni = stof(currentLine[1]);
      else if (pref == "d")
        d = stof(currentLine[1]);
      else if (pref == "pbrboost")
        pbrboost = true;
    }
    else if (pref == "noKdOnTex")
      noKdOnTex = true;
  }

  file.close();

  Material mtl;

  switch (illum)
  {
  case 0:
  case 1:
  case 2:
    mtl = Material(kd, ns / 1000.0f, ni, pbrboost ? 5 : 0);
    mtl.ke = ke;
    break;
  case 4:
  case 6:
  case 7:
  case 9:
    mtl = Material(kd, ns / 1000.0f, ni, pbrboost ? 6 : 2);
    break;
  case 3:
  case 8:
    mtl = Material(kd, 0.0f, 4);
    break;
  case 5:
    mtl = Material(kd, ns / 1000, 1);
    break;
  default:
    mtl = Material(Vec3(1.0f, 1.0f, 1.0f), Vec3(0.0f, 0.0f, 0.0f));
    break;
  }
  mtl.d = d;
  if (kdfile.length() > 0 && filesystem::exists("res/textures/" + kdfile))
  {
    if (noKdOnTex)
      mtl.kd = Vec3(1.0, 1.0, 1.0);
    // printf("kd texture is %s, ", kdfile.c_str());
    map<string, int>::iterator it = texvars.find(kdfile);
    if (it != texvars.end())
    {
      mtl.kdtex = it->second;
      // printf("already found at idx %d.\n", it->second);
    }
    else
    {
      textures.push_back(kdfile);
      mtl.kdtex = textures.size() - 1;
      texvars.emplace(kdfile, textures.size() - 1);
      // printf("adding it at idx %d.\n", textures.size() - 1);
    }
  }
  /* TODO restore this functionality */
  if (kefile.length() > 0 && filesystem::exists("res/textures/" + kefile))
  {
    if (noKdOnTex)
      mtl.ke = Vec3(1.0, 1.0, 1.0);
    // printf("ke texture is %s, ", kefile.c_str());
    map<string, int>::iterator it = texvars.find(kefile);
    if (it != texvars.end())
    {
      mtl.ketex = it->second;
      // printf("already found at idx %d.\n", it->second);
    }
    else
    {
      textures.push_back(kefile);
      mtl.ketex = textures.size() - 1;
      texvars.emplace(kefile, textures.size() - 1);
      // printf("adding it at idx %d.\n", textures.size() - 1);
    }
  }
  if (dfile.length() > 0 && filesystem::exists("res/textures/" + dfile))
  {
    // printf("d texture is %s, ", dfile.c_str());
    map<string, int>::iterator it = texvars.find(dfile);
    if (it != texvars.end())
    {
      mtl.d_tex = it->second;
      // printf("already found at idx %d.\n", it->second);
    }
    else
    {
      textures.push_back(dfile);
      mtl.d_tex = textures.size() - 1;
      texvars.emplace(dfile, textures.size() - 1);
      // printf("adding it at idx %d.\n", textures.size() - 1);
    }
  }
  materials.push_back(mtl);
  // printf("Texture summary: kdtex is %d; ketex is %d; d_tex is %d\n", mtl.kdtex, mtl.ketex, mtl.d_tex);

  return materials.size() - 1;
}

bool loadObj(vector<Triangle> &mesh, vector<TriangleData> &triangleData,
             vector<Material> &materials, vector<string> &textures,
             string model, int m, Vec3 pos, Vec3 scl)
{
  printf("%lu\n", sizeof(Material));
  ifstream file("res/models/" + model + ".obj");
  if (!file.is_open())
    return false;

  bool open = false;
  string objname;
  string line;

  map<string, int> mtlvars;
  map<string, int> texvars;
  int currentMtl = m;
  vector<Vec3> vertices;
  vector<Vec3> textureVertices;
  vector<Vec3> normalVertices;

  while (getline(file, line))
  {
    string pref = firstToken(line);
    vector<string> currentLine;
    split(line, currentLine, " ");
    if (pref == "usemtl")
    {
      // Don't load twice (takes up more space)
      map<string, int>::iterator it = mtlvars.find(currentLine[1]);
      if (it != mtlvars.end())
      {
        currentMtl = it->second;
      }
      else
      {
        currentMtl = loadMtl(materials, textures, model, currentLine[1], texvars,
                             currentMtl);
        // printf("new mat: %s, idx: %d\n", currentLine[1].c_str(), currentMtl);
        mtlvars.emplace(currentLine[1], currentMtl);
      }
    }
    else if (pref == "v")
    {
      vector<string> spos;
      Vec3 vpos;
      split(tail(line), spos, " ");

      vpos.x = stof(currentLine[1]);
      vpos.y = stof(currentLine[2]);
      vpos.z = stof(currentLine[3]);

      vpos = vpos * scl + pos;

      vertices.push_back(vpos);
    }
    else if (pref == "vn")
    {
      vector<string> spos;
      Vec3 vpos;
      split(tail(line), spos, " ");

      vpos.x = stof(currentLine[1]);
      vpos.y = stof(currentLine[2]);
      vpos.z = stof(currentLine[3]);

      normalVertices.push_back(vpos);
    }
    else if (pref == "vt")
    {
      vector<string> spos;
      Vec3 vpos;
      split(tail(line), spos, " ");

      vpos.x = stof(currentLine[1]);
      vpos.y = stof(currentLine[2]);

      textureVertices.push_back(vpos);
    }
    else if (pref == "f")
    {
      if (currentLine.size() == 4)
      {
        // printf("tri/quad mat idx: %d\n", currentMtl);
        vector<string> sfce1;
        split(currentLine[1], sfce1, "/");
        vector<string> sfce2;
        split(currentLine[2], sfce2, "/");
        vector<string> sfce3;
        split(currentLine[3], sfce3, "/");

        Vec3 v0 = vertices[stoi(sfce1[0]) - 1];
        Vec3 v1 = vertices[stoi(sfce2[0]) - 1];
        Vec3 v2 = vertices[stoi(sfce3[0]) - 1];

        Triangle tfce = makeTriangle(v0, v1, v2, currentMtl);

        mesh.push_back(tfce);

        TriangleData data;

        if (sfce1.size() > 1 && sfce1[1].length() > 0 && sfce2.size() > 1 && sfce2[1].length() > 0 && sfce3.size() > 2 && sfce3[1].length() > 0)
        {
          data.t0 = Vec3(textureVertices[stoi(sfce1[1]) - 1], 1);
          data.t1 = Vec3(textureVertices[stoi(sfce2[1]) - 1], 1);
          data.t2 = Vec3(textureVertices[stoi(sfce3[1]) - 1], 1);
        }
        else
        {
          data.t0.w = 0;
          data.t1.w = 0;
          data.t2.w = 0;
        }

        if (sfce1.size() > 2 && sfce1[2].length() > 0 && sfce2.size() > 2 && sfce2[2].length() > 0 && sfce3.size() > 2 && sfce3[2].length() > 0)
        {
          data.n0 = Vec3(normalVertices[stoi(sfce1[2]) - 1], 1);
          data.n1 = Vec3(normalVertices[stoi(sfce2[2]) - 1], 1);
          data.n2 = Vec3(normalVertices[stoi(sfce3[2]) - 1], 1);
        }
        else
        {
          data.n0.w = 0;
          data.n1.w = 0;
          data.n2.w = 0;
        }

        float det = dot(v0, cross(v1, v2));

        data.uv = cross(v1, v2) / det;
        data.vv = cross(v0, v2) / (-det);
        // // printf("det: %f, uv: (%f, %f, %f), vv: (%f, %f, %f)\n", det, data.uv.x, data.uv.y, data.uv.z, data.vv.x, data.vv.y, data.vv.z);

        triangleData.push_back(data);
      }
      else
      {
        vector<string> sfce1;
        split(currentLine[1], sfce1, "/");
        vector<string> sfce2;
        split(currentLine[2], sfce2, "/");
        vector<string> sfce3;
        split(currentLine[3], sfce3, "/");

        Vec3 v0 = vertices[stoi(sfce1[0]) - 1];
        Vec3 v1 = vertices[stoi(sfce2[0]) - 1];
        Vec3 v2 = vertices[stoi(sfce3[0]) - 1];

        Triangle tfce = makeTriangle(v1, v2, v0, currentMtl);

        mesh.push_back(tfce);

        TriangleData data;

        if (sfce1.size() > 1 && sfce1[1].length() > 0 && sfce2.size() > 1 && sfce2[1].length() > 0 && sfce3.size() > 2 && sfce3[1].length() > 0)
        {
          data.t0 = Vec3(textureVertices[stoi(sfce2[1]) - 1], 1);
          data.t1 = Vec3(textureVertices[stoi(sfce3[1]) - 1], 1);
          data.t2 = Vec3(textureVertices[stoi(sfce1[1]) - 1], 1);
        }
        else
        {
          data.t0.w = 0;
          data.t1.w = 0;
          data.t2.w = 0;
        }

        if (sfce1.size() > 2 && sfce1[2].length() > 0 && sfce2.size() > 2 && sfce2[2].length() > 0 && sfce3.size() > 2 && sfce3[2].length() > 0)
        {
          data.n0 = Vec3(normalVertices[stoi(sfce2[2]) - 1], 1);
          data.n1 = Vec3(normalVertices[stoi(sfce3[2]) - 1], 1);
          data.n2 = Vec3(normalVertices[stoi(sfce1[2]) - 1], 1);
        }
        else
        {
          data.n0.w = 0;
          data.n1.w = 0;
          data.n2.w = 0;
        }

        float det = dot(v0, cross(v1, v2));

        data.uv = cross(v1, v2) / det;
        data.vv = cross(v0, v2) / (-det);
        // // printf("det: %f, uv: (%f, %f, %f), vv: (%f, %f, %f)\n", det, data.uv.x, data.uv.y, data.uv.z, data.vv.x, data.vv.y, data.vv.z);

        triangleData.push_back(data);
        vector<string> sfce4;
        split(currentLine[4], sfce4, "/");

        Vec3 v3 = vertices[stoi(sfce4[0]) - 1];

        Triangle tfce2 = makeTriangle(v0, v2, v3, m);

        mesh.push_back(tfce2);

        TriangleData data2;

        if (sfce1.size() > 1 && sfce1[1].length() > 0 && sfce3.size() > 1 && sfce3[1].length() > 0 && sfce4.size() > 1 && sfce4[1].length() > 0)
        {
          data2.t0 = Vec3(textureVertices[stoi(sfce1[1]) - 1], 1);
          data2.t1 = Vec3(textureVertices[stoi(sfce3[1]) - 1], 1);
          data2.t2 = Vec3(textureVertices[stoi(sfce4[1]) - 1], 1);
        }
        else
        {
          data2.t0.w = 0;
          data2.t1.w = 0;
          data2.t2.w = 0;
        }

        if (sfce1.size() > 2 && sfce1[2].length() > 0 && sfce3.size() > 2 && sfce3[2].length() > 0 && sfce4.size() > 2 && sfce4[2].length() > 0)
        {
          data2.n0 = Vec3(normalVertices[stoi(sfce1[2]) - 1], 1);
          data2.n1 = Vec3(normalVertices[stoi(sfce3[2]) - 1], 1);
          data2.n2 = Vec3(normalVertices[stoi(sfce4[2]) - 1], 1);
        }
        else
        {
          data2.n0.w = 0;
          data2.n1.w = 0;
          data2.n2.w = 0;
        }

        float det2 = dot(v0, cross(v2, v3));

        data2.uv = cross(v2, v3) / det2;
        data2.vv = cross(v0, v3) / (-det2);
        // // printf("det: %f, uv: (%f, %f, %f), vv: (%f, %f, %f)\n", det, data.uv.x, data.uv.y, data.uv.z, data.vv.x, data.vv.y, data.vv.z);

        triangleData.push_back(data2);
      }
    }
  }

  file.close();

  return true;
}
