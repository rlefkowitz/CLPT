#pragma once

#include "primitives.h"
#include "parser.h"

using namespace std;

bool loadObj(vector<Triangle> &mesh, string path, int m, Vec3 pos, Vec3 scl) {
    vector<Vec3> vertices;
    ifstream file("res/" + path + ".obj");
    if(!file.is_open())
        return false;
    
    vertices.clear();
    
    bool open = false;
    string objname;
    
    string currentLine;
    while(getline(file, currentLine)) {
        if(algorithm::firstToken(currentLine) == "v") {
            vector<string> spos;
            Vec3 vpos;
            algorithm::split(algorithm::tail(currentLine), spos, " ");
            
            vpos.x = stof(spos[0]);
            vpos.y = stof(spos[1]);
            vpos.z = stof(spos[2]);
            
            vpos = vpos * scl + pos;
            
            vertices.push_back(vpos);
            
//            cout << "[" << vpos.x << ", " << vpos.y << ", " << vpos.z << "]" << endl;
            
        }
        else if(algorithm::firstToken(currentLine) == "f") {
            vector<string> sfce;
            algorithm::split(algorithm::tail(currentLine), sfce, " ");
            vector<string> sfce1;
            algorithm::split(sfce[0], sfce1, "/");
            vector<string> sfce2;
            algorithm::split(sfce[1], sfce2, "/");
            vector<string> sfce3;
            algorithm::split(sfce[2], sfce3, "/");
            
//            cout << "v" << sfce1[0] << " v" << sfce2[0] << " v" << sfce3[0] << endl;
            
            Vec3 v0 = vertices[stoi(sfce1[0]) - 1];
            Vec3 v1 = vertices[stoi(sfce2[0]) - 1];
            Vec3 v2 = vertices[stoi(sfce3[0]) - 1];
            
            Triangle tfce = makeTriangle(v0, v1, v2, m);
            
            mesh.push_back(tfce);

        }
    }
    
    file.close();
    
    return true;
    
    
}
