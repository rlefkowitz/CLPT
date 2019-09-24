#pragma once

#include "geometry.h"

using namespace std;

namespace algorithm
{
    
    // Split a String into a string array at a given token
    inline void split(const string &in,
                      vector<string> &out,
                      string token)
    {
        out.clear();
        
        string temp;
        
        for (int i = 0; i < int(in.size()); i++)
        {
            string test = in.substr(i, token.size());
            
            if (test == token)
            {
                if (!temp.empty())
                {
                    out.push_back(temp);
                    temp.clear();
                    i += (int)token.size() - 1;
                }
                else
                {
                    out.push_back("");
                }
            }
            else if (i + token.size() >= in.size())
            {
                temp += in.substr(i, token.size());
                out.push_back(temp);
                break;
            }
            else
            {
                temp += in[i];
            }
        }
    }
    
    // Get tail of string after first token and possibly following spaces
    inline string tail(const string &in)
    {
        size_t token_start = in.find_first_not_of(" \t");
        size_t space_start = in.find_first_of(" \t", token_start);
        size_t tail_start = in.find_first_not_of(" \t", space_start);
        size_t tail_end = in.find_last_not_of(" \t");
        if (tail_start != string::npos && tail_end != string::npos)
        {
            return in.substr(tail_start, tail_end - tail_start + 1);
        }
        else if (tail_start != string::npos)
        {
            return in.substr(tail_start);
        }
        return "";
    }
    
    // Get first token of string
    inline string firstToken(const string &in)
    {
        if (!in.empty())
        {
            size_t token_start = in.find_first_not_of(" \t");
            size_t token_end = in.find_first_of(" \t", token_start);
            if (token_start != string::npos && token_end != string::npos)
            {
                return in.substr(token_start, token_end - token_start);
            }
            else if (token_start != string::npos)
            {
                return in.substr(token_start);
            }
        }
        return "";
    }
    
    // Get element at given index position
    template <class T>
    inline const T & getElement(const vector<T> &elements, string &index)
    {
        int idx = stoi(index);
        if (idx < 0)
            idx = int(elements.size()) + idx;
        else
            idx--;
        return elements[idx];
    }
}


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
