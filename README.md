# CLPT: An Interactive OpenCL-based Path Tracing Engine
## Created and Developed by Ryan Lefkowitz

![alt text](https://lh3.googleusercontent.com/dY6H78O1iMj6yvAftwHIeZ8JCXfsBDFn0aXAhUQeyqgP-Hgzd8xXHDrmRWl6T45qLMDuP67upJSMioQQn0lM5xQiWe9YdrITUAfhpuQtlWljy6RZPbudtpvuxI7dzNpSxVPATBsvgCJVxpN8_jkSk2__PnOt0W-mssXhLcMckSh4rR6aFbm3Fvs1LhXd07Hds6xxfyMbKfwOTW4jWveJwWXLpwImohAKXflK3rSBZAy4Qn6GwOnvESpVemozvDPfDXKkY2-s9oUsAmcXqH8CJf6RRk4La6cexYZlJdhKeouMYBjCA8s51mDBnBgJOSSDVN_YZDe-QaZSpM3sRNbe_xLPz455HRN25WQfZEDUhzlVr_emqB3_RJwaDycdkB23WWkc9jMQISmNiB2bMbv_3lmVyUqgQJ0Os9yNr4nwNZPJUrW9V7mfuqra95a2SR6rnvCOEMJyTje3EWNNTbHG9W4lyC_So46zRwkUAAbkwrAwRWLq88ss9BZCgn5-cDUVrpHCBomr2GjnNNwL-yhnrnAHt3kRuREWWZCO1oDd1UTxcryAe3ceDcpvqTgqW0-n58t4WIuKNvyExZSbr3Wa8OYSATdJrTHf25lzZA5tThnpvIQ8ZRSp9a0N4mX20xnV3VgRznZT4DIvKxjWgUqh1GafSyobq5_Ngai4BLIgZDeJK3X0118isyQ=w1920-h1080-no "Demo Scene (1920x1080, 20,000+ SPP), Rendered in Under 10 Minutes")

## Features
- Monte Carlo Path Tracing
- Real-time rendering speed
- SAH BVH (Surface Area Heuristic Bounding Volume Hierarchy) acceleration structure for meshes
- Progressive rendering
- Depth of Field (with pentagonal bokeh) based on the thin-lens model
- Image-Based Lighting using an HDR environment map

### Supported primitives
- Spheres
- Triangle Meshes
- Ground Plane

### Supported materials
- Lambertian diffuse
- GGX Plastic
- GGX Dielectric
- Mirror

\**all materials can be emissive*


## How to use CLPT
Run the file `run.command` to start the engine. This will build, compile, and execute the program. You can then choose from the OpenCL-capable devices available on your machine (this project is currently a WIP, so not all hardware will work). 

After a bit of initialization, the demo scene will load and the display will appear.

### Motion Controls
- W/S to go forward/backward
- D/A to go right/left
- R/F to go up/down
- click and move mouse to look around

### Camera Controls
- G/H to increase/decrease the aperture radius
- T/Y to increase/decrease the focal distance

### Other Controls
- SPACE to reset position and camera attributes
- ESC to Quit

Thank you for checking out CLPT!


## Planned Future Changes

### Optimizations
Further optimizations could be implemented to improve the speed and efficiency of CLPT
#### Acceleration structures
- SBVH (Stich et al. 2009) has been shown to have an identical worst-case performance to the SAH BVH, while boasting a significant increase in best-case performance
#### GPU utlization
- Minimize branching to reduce execution divergence
- Optimize memory alignment
- Use local memory more often
  - Currently using global memory to store geometry data and constant memory to store material data
  - Local memory has been shown to be accessible up to 100x faster than global memory, though is certainly limited in size
- Reduce Idle Threads
  - It is likely that with local group sizes larger than several pixels, many threads will escape the scene early or terminate early due to Russian Roulette
    - As a result, these threads are idling, which will worsen the consequences of execution divergence
#### Should explore Wavefront Path Tracing (Laine et al. 2013)
-Better streamlines the path tracing pipeline for GPU usage

### Features
There are many features which could be implemented to increase the user experience, photorealism, and capabilities of CLPT
#### Volumetric Path Tracing
- Would improve realism of scenes and make available many complex materials
- Though it would certainly cause some additional latency and high-frequency noise would be more prevalent
#### Conductive materials (i.e. Metal)
- Certainly feasible for the renderer
