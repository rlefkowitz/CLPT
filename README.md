# CLPT: An Interactive OpenCL-based Monte Carlo Path Tracing Engine
## Created and Developed by Ryan Lefkowitz

![Alt text](https://raw.githubusercontent.com/rlefkowitz/CLPT/master/CoverImage.png)
The scene `clpt.clpt` (1280x640, 51,000+ SPP), rendered in just over an hour.

## Features
- Monte Carlo Path Tracing
- Real-time rendering speed
- SAH BVH (Surface Area Heuristic Bounding Volume Hierarchy) acceleration structure for meshes
- Progressive rendering
- Depth of Field (with pentagonal bokeh) based on the thin-lens model
- Image-Based Lighting using an HDR environment map
- Volumetric Path Tracing with homogeneous media using full scattering
- Custom, readable scene (`.clpt`) and configuration (`.clptcfg`) file formats

### Supported primitives
- Spheres
- Triangle Meshes
- Ground Plane

### Supported materials
- Lambertian diffuse
- GGX Plastic
- GGX Dielectric/Glass
- Mirror

\**all materials can be emissive*

\**all materials can use a homogenous Isotropic Scattering Medium (only transparent materials will be affected, however)*


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

### Configuration Files
CLPT's configuration files, specified with a `.clptcfg` or `.clptcfg.txt` extension, are CLPT's way of allowing the user to specify basic parameters of the simulation, such as resolution, the name of the scene file to be used, and whether the scene is interactive (the following example is interactive, as long as the first word of a line is not `interactive` interactivity will be disabled).
```
width: 1920
height: 1080
scene: default
interactive
```

### Scene Files
CLPT's scene files, specified with a `.clpt` or `.clpt.txt` extension, are CLPT's way of allowing the user to specify all aspects of a scene.

Here's the scene file for `default.clpt`:
```
 # set camera attributes
setCameraPosition 4.216578948221484 2.375 0.34339889486771863
setCameraForward -0.7156478248575902 -0.05930652721420354 0.6959388813727766
setCameraUp -0.042517570286490336 0.9982398187959389 0.04134634672114762
setCameraFocalDistance 7.370589916300956
setCameraApertureRadius 5e-2

 # toggle options
enableIbL
enableDOF
enableGround

 # specify IbL path and set IbL weights by component (or background color if no IbL)
iblPath res/Frozen_Waterfall_Ref.hdr
backgroundColor 1.0 1.0 1.0

 # set a vector type variable 'blue' to (0.3, 0.3, 0.9)
set vec3 blue 0.3 0.3 0.9

 # create a material 'bluePlastic', a plastic material with a diffuse color of 'blue' and a roughness of 0.103'
material bluePlastic plastic blue 0.103

 # add a deer mesh to the scene at position (-3, 0, 5) with scale (0.25, 0.25, 0.25) using the 'bluePlastic' material
mesh deer -3 0 5 0.25 0.25 0.25 bluePlastic
```
For a longer, more complex example, go to the scene file used to produce the image just below the title of this page at `res/clpt.clpt`.



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
- Better streamlines the path tracing pipeline for GPU usage


### Features
There are many features which could be implemented to increase the user experience, photorealism, and capabilities of CLPT

#### Conductive materials (i.e. Metal)
- Certainly feasible for the renderer
