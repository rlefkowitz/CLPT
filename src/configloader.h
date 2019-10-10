#pragma once

#include "parser.h"

using namespace std;

bool loadConfig(string path, int* width, int* height, string* scn_name, bool* interactive) {
    *width = 1280;
    *height = 720;
    *scn_name = "default";
    *interactive = false;
    
    ifstream file("res/" + path + ".clptcfg");
    if(!file.is_open()) {
        file = ifstream("res/" + path + ".clptcfg.txt");
        if(!file.is_open())
            return false;
    }
    
    bool open = false;
    
    string currentLine;
    while(getline(file, currentLine)) {
        string first = algorithm::firstToken(currentLine);
        transform(first.begin(), first.end(), first.begin(), ::tolower);
        if(first.at(first.size() - 1) == ':')
            first.pop_back();
        string rest = algorithm::tail(currentLine);
        if(first == "width") {
            try {
                *width = stoi(rest);
            } catch(int e) {
                printf("Error: invalid width entered!");
                exit(1);
            }
        } else if(first == "height") {
            try {
                *height = stoi(rest);
            } catch(int e) {
                printf("Error: invalid height entered!");
                exit(1);
            }
        } else if(first == "scene")
            *scn_name = rest;
        else if(first == "interactive")
            *interactive = true;
        
    }
    
    file.close();
    
    return true;
    
}
