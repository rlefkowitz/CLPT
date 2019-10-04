# CLPT: An Interactive OpenCL-based Path Tracing Engine
## Created and Developed by Ryan Lefkowitz


## Features
-Monte Carlo Path Tracing
-Real-time rendering speed
-SAH BVH (Surface Area Heuristic Bounding Volume Hierarchy) acceleration structure for meshes
-Progressive rendering
-Depth of Field (with pentagonal bokeh) based on the thin-lens model
-Image-Based Lighting using an HDR environment map

### Supported primitives
-Spheres
-Triangle Meshes
-Ground Plane

### Supported materials
-Lambertian diffuse
-GGX Plastic
-GGX Dielectric
-Mirror
\**all materials can be emissive*


## How to use CLPT
Run the file `run.command` to start the engine. This will build, compile, and execute the program. You can then choose from the OpenCL-capable devices available on your machine (this project is currently a WIP, so not all hardware will work). 

After a bit of initialization, the demo scene will load and the display will appear.

### Motion Controls
-W/S to go forward/backward
-D/A to go right/left
-R/F to go up/down
-click and move mouse to look around

### Camera Controls
-G/H to increase/decrease the aperture radius
-T/Y to increase/decrease the focal distance

### Other Controls
-SPACE to reset position and camera attributes
-ESC to Quit

Thank you for checking out CLPT!


## Planned Future Changes

### Optimizations
Further optimizations could be implemented to improve the speed and efficiency of CLPT
#### Acceleration structures
-SBVH (Stich et al. 2009) has been shown to have an identical worst-case performance to the SAH BVH, while boasting a significant increase in best-case performance
#### GPU utlization
-Minimize branching to reduce execution divergence
-Optimize memory alignment
-Use local memory more often
  -Currently using global memory to store geometry data and constant memory to store material data
  -Local memory has been shown to be accessible up to 100x faster than global memory, though is certainly limited in size
-Reduce Idle Threads
  -It is likely that with local group sizes larger than several pixels, many threads will escape the scene early or terminate early due to Russian Roulette
    -As a result, these threads are idling, which will certainly worsen the consequences of execution divergence
#### Should explore Wavefront Path Tracing (Laine et al. 2013)
-Better streamlines the path tracing pipeline for GPU usage

### Features
There are many features which could be implemented to increase the user experience, photorealism, and capabilities of CLPT
#### Volumetric Path Tracing
-Would improve realism of scenes and make available many complex materials
-Though it would certainly cause some additional latency and high-frequency noise would be more prevalent
#### Conductive materials (i.e. Metal)
-Certainly feasible for the renderer
