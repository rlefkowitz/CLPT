CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

PROJECT ( CLPath-V0.0.1 )

# Define to silence deprecation warnings
add_definitions(-DGL_SILENCE_DEPRECATION)

# Set CXX Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

# Include directories for frameworks
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework OpenCL")

#########################################################
# FIND GLUT (FreeGLUT)
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
link_directories(${GLUT_LIBRARY_DIR})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
link_directories(${OPENGL_LIBRARY_DIR})
add_definitions(${OPENGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

#########################################################
# FIND OPENCL
#########################################################
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY_DIRS})
add_definitions(${OpenCL_DEFINITIONS})
if(NOT OpenCL_FOUND)
    message(ERROR " OPENCL not found!")
endif(NOT OpenCL_FOUND)

#########################################################
# FIND PNG
#########################################################
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
link_directories(${PNG_LIBRARY_DIRS})
add_definitions(${PNG_DEFINITIONS})
if(NOT PNG_FOUND)
    message(ERROR " PNG not found!")
endif(NOT PNG_FOUND)

SET ( CLPATH_SOURCE
  src/camera.cpp
  src/hdrloader/hdrloader.cpp
  )

SET ( CLPATH_HEADERS
  src/bvh.h
  src/camera.h
  src/chunk.h
  src/cl_gl_interop.h
  src/configloader.h
  src/hdrloader/hdrloader.h
  src/material.h
  src/objloader.h
  src/parser.h
  src/primitives.h
  src/scene.h
  src/sceneloader.h
  src/spice.h
  src/user_interaction.h
  src/vec.h
  )

ADD_LIBRARY(clpath STATIC ${CLPATH_SOURCE} ${CLPATH_HEADERS})

ADD_EXECUTABLE(clpath_exe src/main.cpp)

TARGET_LINK_LIBRARIES(clpath_exe clpath ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${OpenCL_LIBRARY} ${PNG_LIBRARY})

SET_TARGET_PROPERTIES ( clpath_exe
  PROPERTIES
  OUTPUT_NAME clpath
  )

INSTALL ( TARGETS
  clpath
  DESTINATION
  bin
  )

INSTALL ( TARGETS
  clpath
  DESTINATION
  lib
  )
