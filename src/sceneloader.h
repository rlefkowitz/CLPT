#pragma once

#include <map>
#include <sstream>
#include <string>
#include "parser.h"
#include "scene.h"

using namespace std;

/*
 Credit to https://stackoverflow.com/questions/2844817/how-do-i-check-if-a-c-string-is-an-int
 */
inline bool isInteger(const std::string & s)
{
    if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false;
    
    char * p;
    strtol(s.c_str(), &p, 10);
    
    return (*p == 0);
}

/*
 Credit to https://stackoverflow.com/questions/447206/c-isfloat-function
 */
bool isFloat( string myString ) {
    std::istringstream iss(myString);
    float f;
    iss >> noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail();
}

int getIntVar(map<string, int> intvars, string varName) {
    map<string, int>::iterator it = intvars.find(varName);
    if(it != intvars.end()) {
        return it->second;
    } else {
        cout << "Error: " << varName << "\" was never declared!" << endl;
        exit(1);
    }
    return -1;
}

float getFloatVar(map<string, float> floatvars, string varName) {
    map<string, float>::iterator it = floatvars.find(varName);
    if(it != floatvars.end()) {
        return it->second;
    } else {
        cout << "Error: " << varName << "\" was never declared!" << endl;
        exit(1);
    }
    return -1;
}

Vec3 getVecVar(map<string, Vec3> vecvars, string varName) {
    map<string, Vec3>::iterator it = vecvars.find(varName);
    if(it != vecvars.end()) {
        return it->second;
    } else {
        cout << "Error: " << varName << "\" was never declared!" << endl;
        exit(1);
    }
    return -1;
}

int extractInt(vector<string> &info, map<string, int> intvars) {
    int val;
    if(isInteger(info[0])) {
        val = stoi(info[0]);
    } else {
        val = getIntVar(intvars, info[0]);
    }
    info.erase(info.begin());
    return val;
}

float extractFloat(vector<string> &info, map<string, float> floatvars) {
    float val;
    if(isFloat(info[0])) {
        val = stof(info[0]);
    } else {
        val = getFloatVar(floatvars, info[0]);
    }
    info.erase(info.begin());
    return val;
}

Vec3 extractVec3(vector<string> &info, map<string, Vec3> vecvars, map<string, float> floatvars) {
    bool extractAsVar = false;
//    cout << info[0] << endl;
    if(isFloat(info[0])) {
        stof(info[0]);
    } else {
        Vec3 var = getVecVar(vecvars, info[0]);
        info.erase(info.begin());
        extractAsVar = true;
        return var;
    }
    float v[3];
    for(int i = 0; i < 3; i++)
        v[i] = extractFloat(info, floatvars);
    return Vec3(v[0], v[1], v[2]);
}

bool loadScene(Scene &scn, string path) {
//    cout << "Got to scene builder." << endl;
    map<string, int> intvars;
    map<string, float> floatvars;
    map<string, Vec3> vecvars;
//    cout << "Allocated maps." << endl;
    bool IbL_enabled = false;
    bool DOF_enabled = false;
    bool use_ground = false;
    Vec3 cam_pos = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 cam_fd = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 cam_up = Vec3(0.0f, 1.0f, 0.0f);
    float cam_focal_distance = 10.0f;
    float cam_aperture_radius = 0.0f;
//    cout << "Set initial camera and bool values." << endl;
    vector<Sphere> spheres;
    vector<Triangle> triangles;
    vector<BVHNode> nodes;
    vector<Material> materials;
    vector<Medium> mediums;
//    cout << "Created vector lists." << endl;
    string iblPath = "";
    Vec3 background_color = Vec3(1.0f, 1.0f, 1.0f);
//    cout << "Created background stuff." << endl;
    
//    cout << "Allocated all relevant init info." << endl;
    
    
    ifstream file("res/" + path + ".clpt");
    if(!file.is_open()) {
        file = ifstream("res/" + path + ".clpt.txt");
        if(!file.is_open())
            return false;
    }
    
    bool open = false;
    
    string currentLine;
    while(getline(file, currentLine)) {
        string first = algorithm::firstToken(currentLine);
        vector<string> info;
        algorithm::split(algorithm::tail(currentLine), info, " ");
        if(first == "set") {
            string varType = algorithm::firstToken(info[0]);
            string varName = algorithm::firstToken(info[1]);
            info.erase(info.begin(), info.begin() + 2);
            
            if(varType == "float") {
                floatvars.emplace(varName, extractFloat(info, floatvars));
            } else if(varType == "vec3") {
                vecvars.emplace(varName, extractVec3(info, vecvars, floatvars));
            } else if(varType == "int") {
                intvars.emplace(varName, extractInt(info, intvars));
            }
            
        } else if(first == "material") {
            string varName = algorithm::firstToken(info[0]);
            info.erase(info.begin());
            Material mtl;
            Vec3 ke = Vec3(0.0f, 0.0f, 0.0f);
            if(info[0] == "emit") {
                info.erase(info.begin());
                ke = extractVec3(info, vecvars, floatvars);
            }
            string matType = info[0];
            info.erase(info.begin());
            Vec3 kd = extractVec3(info, vecvars, floatvars);
            int medIdx = -1;
//            cout << "Created ";
            if(matType == "diffuse") {
                mtl = Material(kd);
//                cout << "diffuse material with albedo (" << kd.x << ", " << kd.y << ", " << kd.z << ") called " << varName << "." << endl;
            } else if(matType == "plastic") {
                float roughness = extractFloat(info, floatvars);
                mtl = Material(kd, roughness, 1);
            } else if(matType == "dielectric" || matType == "glass") {
                float roughness = extractFloat(info, floatvars);
                float ior = extractFloat(info, floatvars);
                mtl = Material(kd, roughness, ior, 2);
            } else if(matType == "metal") {
                Vec3 k = extractVec3(info, vecvars, floatvars);
                float roughness = extractFloat(info, floatvars);
                mtl = Material(kd, k, roughness, 3);
            } else if(matType == "mirror") {
                mtl = Material(kd, 0.0f, 4);
            }
            if(info.size() > 0) {
                if(info[0] == "medium") {
                    info.erase(info.begin());
                }
                medIdx = extractInt(info, intvars);
            }
            if(matType != "metal")
                mtl.ke = ke;
            mtl.medIdx = medIdx;
            
            int idx = materials.size();
            materials.push_back(mtl);
            intvars.emplace(varName, idx);
        } else if(first == "medium") {
            string varName = algorithm::firstToken(info[0]);
            info.erase(info.begin());
            Medium med;
            if(info.size() == 2) {
                float f0 = extractFloat(info, floatvars);
                float f1 = extractFloat(info, floatvars);
                med = Medium(f0, f1);
            } else if(info.size() == 3) {
                float f0 = extractFloat(info, floatvars);
                float f1 = extractFloat(info, floatvars);
                float f2 = extractFloat(info, floatvars);
                med = Medium(f0, f1, f2);
            } else {
                Vec3 v = extractVec3(info, vecvars, floatvars);
                float f0 = extractFloat(info, floatvars);
                if(info.size() > 0) {
                    float f1 = extractFloat(info, floatvars);
                    med = Medium(v, f0, f1);
                } else {
                    med = Medium(v, f0);
                }
            }
            int idx = mediums.size();
            mediums.push_back(med);
            intvars.emplace(varName, idx);
        } else if(first == "sphere") {
            float radius = extractFloat(info, floatvars);
            Vec3 pos = extractVec3(info, vecvars, floatvars);
            int mtlIdx = extractInt(info, intvars);
            Sphere sphereToAdd(radius, pos, mtlIdx);
            spheres.push_back(sphereToAdd);
        } else if(first == "mesh") {
            string model = info[0];
            info.erase(info.begin());
            Vec3 pos = extractVec3(info, vecvars, floatvars);
            Vec3 scl = extractVec3(info, vecvars, floatvars);
            int mtlIdx = extractInt(info, intvars);
            loadObj(triangles, model, mtlIdx, pos, scl);
        } else if(first == "setCameraPosition") {
            cam_pos = extractVec3(info, vecvars, floatvars);
        } else if(first == "setCameraForward") {
            cam_fd = extractVec3(info, vecvars, floatvars);
        } else if(first == "setCameraUp") {
            cam_up = extractVec3(info, vecvars, floatvars);
        } else if(first == "setCameraFocalDistance") {
            cam_focal_distance = extractFloat(info, floatvars);
        } else if(first == "setCameraApertureRadius") {
            cam_aperture_radius = extractFloat(info, floatvars);
        } else if(first == "iblPath") {
            iblPath = info[0];
        } else if(first == "enableIbL") {
            IbL_enabled = true;
        } else if(first == "disableIbL") {
            IbL_enabled = false;
        } else if(first == "enableDOF") {
            DOF_enabled = true;
        } else if(first == "disableDOF") {
            DOF_enabled = false;
        } else if(first == "enableGround") {
            use_ground = true;
        } else if(first == "disableGround") {
            use_ground = false;
        } else if(first == "backgroundColor") {
            background_color = extractVec3(info, vecvars, floatvars);
        }
    }
    
    file.close();
    
    scn.spheres = spheres;
    scn.triangles = triangles;
    scn.nodes = nodes;
    scn.materials = materials;
    scn.mediums = mediums;
    scn.cam_pos = cam_pos;
    scn.cam_fd = cam_fd;
    scn.cam_up = cam_up;
    scn.cam_focal_distance = cam_focal_distance;
    scn.cam_aperture_radius = cam_aperture_radius;
//    scn.interactiveCamera = new InteractiveCamera(cam_pos, cam_fd, cam_up, cam_focal_distance, cam_aperture_radius);
//    cout << "Created camera with position (" << cam_pos.x << ", " << cam_pos.y << ", " << cam_pos.z << "), forward vec (" << cam_fd.x << ", " << cam_fd.y << ", " << cam_fd.z << "), up vec ("  << cam_up.x << ", " << cam_up.y << ", " << cam_up.z << "), focal distance " << cam_focal_distance << ", and aperture radius " << cam_aperture_radius << "." << endl;
    scn.iblPath = iblPath;
    scn.background_color = background_color;
    scn.use_IbL = IbL_enabled;
    scn.use_ground = use_ground;
    scn.use_DOF = DOF_enabled;
    
    return true;
    
    
}
