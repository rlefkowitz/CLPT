#pragma once

#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include "camera.h"

bool save_image = false;

bool buffer_reset = true;
bool interactive;
InteractiveCamera* interactiveCamera;

void initCamera(); // prototype

// keyboard interaction
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) {
            
        case(27) : exit(0);
        case('p') : save_image = true; break;
    }
    if(interactive) {
        buffer_reset = true;
        switch (key) {
            case(' ') : initCamera(); break;
            case('a') : interactiveCamera->strafe(0.2f); break;
            case('d') : interactiveCamera->strafe(-0.2f); break;
            case('r') : interactiveCamera->changeAltitude(0.2f); break;
            case('f') : interactiveCamera->changeAltitude(-0.2f); break;
            case('w') : interactiveCamera->goForward(0.2f); break;
            case('s') : interactiveCamera->goForward(-0.2f); break;
            case('g') : interactiveCamera->changeApertureDiameter(0.1); break;
            case('h') : interactiveCamera->changeApertureDiameter(-0.1); break;
            case('t') : interactiveCamera->changeFocalDistance(0.1); break;
            case('y') : interactiveCamera->changeFocalDistance(-0.1); break;
            default : buffer_reset = false; break;
        }
    }
}

void specialkeys(int key, int, int){
    if(interactive) {
        switch (key) {
                
            case GLUT_KEY_LEFT: interactiveCamera->changeYaw(0.02f); break;
            case GLUT_KEY_RIGHT: interactiveCamera->changeYaw(-0.02f); break;
            case GLUT_KEY_UP: interactiveCamera->changePitch(0.02f); break;
            case GLUT_KEY_DOWN: interactiveCamera->changePitch(-0.02f); break;
                
        }
    }
}

// mouse event handlers

int lastX = 0, lastY = 0;
int theButtonState = 0;
int theModifierState = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;

// camera mouse controls in X and Y direction
void motion(int x, int y)
{
    if(interactive) {
        int deltaX = lastX - x;
        int deltaY = lastY - y;
        
        if (deltaX != 0 || deltaY != 0) {
            
            if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
            {
                interactiveCamera->changeYaw(deltaX * 0.01);
                interactiveCamera->changePitch(-deltaY * 0.01);
            }
            else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
            {
                interactiveCamera->changeAltitude(-deltaY * 0.01);
            }
            
            lastX = x;
            lastY = y;
            buffer_reset = true;
//            glutPostRedisplay();
            
        }
    }
}

void mouse(int button, int state, int x, int y)
{
    theButtonState = button;
    theModifierState = glutGetModifiers();
    lastX = x;
    lastY = y;
    
    motion(x, y);
}
