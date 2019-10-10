#pragma once

using namespace std;

namespace algorithm {
    
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
