#pragma once

#include <OpenCL/cl_gl.h>
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#include "cl.hpp"
#include "user_interaction.h"
#include <time.h>
#include <iostream>

int initial_time = time(NULL), final_time, frame_count = 0;

//#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"   // OpenCL-OpenGL interoperability extension

int window_width = 1920;
int window_height = 1080;

// OpenGL vertex buffer object
GLuint vbo;

// dipslay function prototype
void render();

void initGL(int argc, char** argv){
	// init GLUT for OpenGL viewport
	glutInit(&argc, argv);
	// specify the display mode to be RGB and single buffering
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position
	glutInitWindowPosition(50, 50);
	// specify the initial window size
	glutInitWindowSize(window_width + 1, window_height);
	// create the window and set title
	glutCreateWindow("Basic OpenCL path tracer");

	// register GLUT callback function to display graphics:
	glutDisplayFunc(render);
    
    // functions for user interaction
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialkeys);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    // initialise OpenGL extensions
//    glewInit();

	// initialise OpenGL
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, window_width, 0.0, window_height);
}

void createVBO(GLuint* vbo)
{
	//create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//initialise VBO
	unsigned int size = window_width * window_height * sizeof(cl_float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawGL(){

	//clear all pixels, then render from the vbo
	glClear(GL_COLOR_BUFFER_BIT);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 16, 0); // size (2, 3 or 4), type, stride, pointer
	glColorPointer(4, GL_UNSIGNED_BYTE, 16, (GLvoid*)8); // size (3 or 4), type, stride, pointer

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, window_width * window_height);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	// flip backbuffer to screen
	glutSwapBuffers();
    
    frame_count++;
    final_time = time(NULL);
    if(final_time - initial_time > 0) {
        printf("FPS: %d\n", frame_count / (final_time - initial_time));
        frame_count = 0;
        initial_time = final_time;
    }
}

void Timer(int value) {
	glutPostRedisplay();
	glutTimerFunc(15, Timer, 0); // OpenGL fps limit (1000 ms / 60 fps = 15 ms)
}
