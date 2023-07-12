#define PI 3.141592653589793238f

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <vector>
#include "chunk.h"
#include "cl.hpp"
#include "cl_gl_interop.h"
#include "configloader.h"
#include "hdrloader/hdrloader.h"
#include "sceneloader.h"
#include "user_interaction.h"
#include <png++/png.hpp>

using namespace std;
using namespace cl;

bool OFFLINE = false;

bool profiling = false;
bool wavefront = false;

const int samples = 32768;
const int sample_interval = 128;
const int samplesPerRun = 1;

// Image-based Lighting variables
int ibl_width;
int ibl_height;
cl_float4 *cpu_ibl;

// texture variables
int texture_amt;
int texture_atlas_size;
TextureData *cpu_textureData;
cl_float4 *cpu_textureAtlas;

// Background color
cl_float3 voidcolor;

// Sphere variables
Sphere *cpu_spheres;
int sphere_amt;

// Triangle variables
Triangle *cpu_triangles;
TriangleData *cpu_triangleData;
int triangle_amt;

// BVHNode variables
BVHNode *cpu_bvhs;
int bvhnode_amt;

// Material variables
Material *cpu_materials;
int material_amt;

// Medium variables
Medium *cpu_mediums;
int medium_amt;

// Finished variable
cl_uchar *cpu_finished;

// Actual ID variable
cl_int *cpu_actualIDs;

// Chunk variable
Chunk *cpu_chunks;

// Scene variables
string scn_path;
Scene scn;

// store booleans
unsigned char bools;

// Random generator
default_random_engine generator;
uniform_int_distribution<int> distribution(1, (1 << 31) - 2);

// Camera variable
Camera *cpu_camera = NULL;

// Host (CPU) cl_uint buffer variables
cl_uint *cpu_randoms;
cl_uint *cpu_usefulnums;

// OpenCL boiler plate variables
Device device;
CommandQueue queue;
Kernel kernel;
Kernel init_kernel;
Kernel intersection_kernel;
Kernel intersectionfp_kernel;
Kernel shading_kernel;
Kernel shadingfp_kernel;
Kernel reassign_kernel;
Kernel reassignfp_kernel;
Kernel shift_kernel;
Kernel shift_kernel_alt;
Kernel final_kernel;
Context context;
Program program;

// Device (CPU/GPU) Buffers
Buffer cl_usefulnums;
Buffer cl_mediums;

Buffer cl_globalworkgroupsize;
Buffer cl_chunks;
Buffer cl_chunks_new;
Buffer cl_rays;
Buffer cl_randoms;
Buffer cl_camera;
Buffer cl_actualIDs;
Buffer cl_actualIDs_new;
Buffer cl_finished;
Buffer cl_points;
Buffer cl_normals;
Buffer cl_mtlidxs;
Buffer cl_spheres;
Buffer cl_triangles;
Buffer cl_triangleData;
Buffer cl_nodes;
Buffer cl_textureData;
Buffer cl_textureAtlas;
Buffer cl_accumbuffer;
Buffer cl_throughputs;
Buffer cl_materials;
Buffer cl_ibl;
BufferGL cl_vbo;
vector<Memory> cl_vbos;

// For Offline only
cl_float4 *cpu_output;
Buffer cl_output;

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f
                                                                : x; }

// convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction
inline int toInt(float x) { return int(clamp(x) * 255 + .5); }

void saveImage()
{
  png::image<png::rgb_pixel> image(window_width, window_height);
  int i = 0;
  float *s;
  for (int r = window_height - 1; r >= 0; r--)
    for (int c = 0; c < window_width; c++, i++)
    {
      s = cpu_output[i].s;
      if (s[0] < 0 || s[0] != s[0] || s[1] < 0 || s[1] != s[1] || s[2] < 0 || s[2] != s[2])
      {
        printf("Issue found! (%f, %f, %f)\n", s[0], s[1], s[2]);
      }
      image[r][c] = png::rgb_pixel(toInt(s[0]), toInt(s[1]), toInt(s[2]));
    }
  remove("opencl_raytracer.png");
  image.write("opencl_raytracer.png");
}

#define float3(x, y, z) \
  {                     \
    {                   \
      x, y, z           \
    }                   \
  } // macro to replace ugly initializer braces
unsigned int framenumber = 0;

void pickPlatform(Platform &platform, const vector<Platform> &platforms)
{

  if (platforms.size() == 1)
    platform = platforms[0];
  else
  {
    int input = 0;
    std::cout << "\nChoose an OpenCL platform: ";
    cin >> input;

    // handle incorrect user input
    while (input < 1 || input > platforms.size())
    {
      cin.clear();                               // clear errors/bad flags on cin
      cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
      std::cout << "No such option. Choose an OpenCL platform: ";
      cin >> input;
    }
    platform = platforms[input - 1];
  }
}

void pickDevice(Device &device, const vector<Device> &devices)
{

  if (devices.size() == 1)
    device = devices[0];
  else
  {
    int input = 0;
    std::cout << "\nChoose an OpenCL device: ";
    cin >> input;

    // handle incorrect user input
    while (input < 1 || input > devices.size())
    {
      cin.clear();                               // clear errors/bad flags on cin
      cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
      std::cout << "No such option. Choose an OpenCL device: ";
      cin >> input;
    }
    device = devices[input - 1];
  }
}

void printErrorLog(const Program &program, const Device &device)
{

  // Get the error log and print to console
  string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
  cerr << "Build log:" << std::endl
       << buildlog << std::endl;

  // Print the error log to a file
  FILE *log = fopen("errorlog.txt", "w");
  fprintf(log, "%s\n", buildlog.c_str());
  std::cout << "Error log saved in 'errorlog.txt'" << endl;
  system("PAUSE");
  exit(1);
}

void initOpenCL()
{
  // Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
  vector<Platform> platforms;
  Platform::get(&platforms);
  std::cout << "Available OpenCL platforms : " << endl
            << endl;
  for (int i = 0; i < platforms.size(); i++)
    std::cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;

  // Pick one platform
  Platform platform;
  pickPlatform(platform, platforms);
  std::cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;

  // Get available OpenCL devices on platform
  vector<Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  std::cout << "Available OpenCL devices on this platform: " << endl
            << endl;
  for (int i = 0; i < devices.size(); i++)
  {
    std::cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
    std::cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
    std::cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl
              << endl;
  }

  // Pick one device
  pickDevice(device, devices);
  std::cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl;
  std::cout << "\t\t\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
  std::cout << "\t\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;

  // Create an OpenCL context on that device.
  // Windows specific OpenCL-OpenGL interop
  CGLContextObj kCGLContext = CGLGetCurrentContext();
  CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

  cl_context_properties properties[] = {
      CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
      (cl_context_properties)kCGLShareGroup,
      0};

  // Create a command queue
  context = Context(device, properties);
  std::cout << "Created context\n";
  if (profiling)
  {
    queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
  }
  else
  {
    queue = CommandQueue(context, device);
  }
  std::cout << "Created queue\n";

  // Convert the OpenCL source code to a string// Convert the OpenCL source code to a string
  string source;
  // ifstream file("opencl_kernel_bvhtest.cl");
  ifstream file("opencl_kernel.cl");
  if (!file)
  {
    std::cout << "\nNo OpenCL file found!" << endl
              << "Exiting..." << endl;
    system("PAUSE");
    exit(1);
  }
  while (!file.eof())
  {
    char line[256];
    file.getline(line, 255);
    source += line;
  }

  const char *kernel_source = source.c_str();

  // Create an OpenCL program with source
  program = Program(context, kernel_source);
  std::cout << "Created program\n";

  // Build the program for the selected device
  cl_int result = program.build({device}); // "-cl-fast-relaxed-math"
  std::cout << "Built program on device\n";
  if (result)
    std::cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
  if (result == CL_BUILD_PROGRAM_FAILURE)
    printErrorLog(program, device);
}

void initOpenCL_offline()
{
  // Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
  vector<Platform> platforms;
  Platform::get(&platforms);
  std::cout << "Available OpenCL platforms : " << endl
            << endl;
  for (int i = 0; i < platforms.size(); i++)
    std::cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;

  // Pick one platform
  Platform platform;
  pickPlatform(platform, platforms);
  std::cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;

  // Get available OpenCL devices on platform
  vector<Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  std::cout << "Available OpenCL devices on this platform: " << endl
            << endl;
  for (int i = 0; i < devices.size(); i++)
  {
    std::cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
    std::cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
    std::cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl
              << endl;
  }

  // Pick one device
  pickDevice(device, devices);
  std::cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl;
  std::cout << "\t\t\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
  std::cout << "\t\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;

  // Create a command queue
  context = Context(device);
  std::cout << "Created context\n";
  if (profiling)
  {
    queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
  }
  else
  {
    queue = CommandQueue(context, device);
  }
  std::cout << "Created queue\n";

  // Convert the OpenCL source code to a string// Convert the OpenCL source code to a string
  string source;
  ifstream file("opencl_kernel_offline.cl");
  if (!file)
  {
    std::cout << "\nNo OpenCL file found!" << endl
              << "Exiting..." << endl;
    system("PAUSE");
    exit(1);
  }
  while (!file.eof())
  {
    char line[256];
    file.getline(line, 255);
    source += line;
  }

  const char *kernel_source = source.c_str();

  // Create an OpenCL program with source
  program = Program(context, kernel_source);
  std::cout << "Created program\n";

  // Build the program for the selected device
  cl_int result = program.build({device}); // "-cl-fast-relaxed-math"
  std::cout << "Built program on device\n";
  if (result)
    std::cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
  if (result == CL_BUILD_PROGRAM_FAILURE)
    printErrorLog(program, device);
}

void buildScene()
{

  if (!loadScene(scn, scn_path))
  {
    printf("Error: could not load scene!");
    exit(1);
  }

  vector<Material> materials = scn.materials;
  vector<Medium> mediums = scn.mediums;
  vector<Sphere> spheres = scn.spheres;
  vector<Triangle> triangles = scn.triangles;
  vector<TriangleData> triangleData = scn.triangleData;
  vector<BVHNode> nodes = scn.nodes;
  vector<string> textures = scn.textures;

  /*
     Construct the BVH for all triangles added to the scene, if any.
     */
  if (triangles.size() > 0)
  {
    printf("Building BVH for scene with %d triangles...\n", triangles.size());
    nodes = build(triangles, triangleData);
    printf("BVH finished for scene with %d triangles!\n\n", triangles.size());
  }

  sphere_amt = spheres.size();
  printf("Spheres: %d\n", sphere_amt);

  cpu_spheres = new Sphere[sphere_amt];
  for (int i = 0; i < sphere_amt; i++)
    cpu_spheres[i] = spheres[i];
  spheres.clear();

  triangle_amt = triangles.size();
  printf("Triangles: %d\n", triangle_amt);

  cpu_triangles = new Triangle[triangle_amt];
  cpu_triangleData = new TriangleData[triangle_amt];
  for (int i = 0; i < triangle_amt; i++)
  {
    cpu_triangles[i] = triangles[i];
    cpu_triangleData[i] = triangleData[i];
  }
  triangles.clear();
  triangleData.clear();

  bvhnode_amt = nodes.size();
  printf("BVH Nodes: %d\n", bvhnode_amt);

  cpu_bvhs = new BVHNode[bvhnode_amt];
  for (int i = 0; i < bvhnode_amt; i++)
    cpu_bvhs[i] = nodes[i];
  nodes.clear();

  material_amt = materials.size();
  printf("Materials: %d\n", material_amt);

  cpu_materials = new Material[material_amt];
  for (int i = 0; i < material_amt; i++)
    cpu_materials[i] = materials[i];
  materials.clear();

  medium_amt = mediums.size();
  printf("Mediums: %d\n", medium_amt);

  cpu_mediums = new Medium[medium_amt];
  for (int i = 0; i < medium_amt; i++)
    cpu_mediums[i] = mediums[i];
  mediums.clear();

  texture_amt = textures.size();
  texture_atlas_size = 0;
  cpu_textureData = new TextureData[texture_amt];
  for (int i = 0; i < texture_amt; i++)
  {
    int w, h, n;
    stbi_info(("res/textures/" + textures[i]).c_str(), &w, &h, &n);
    cpu_textureData[i].w = w;
    cpu_textureData[i].h = h;
    cpu_textureData[i].s = texture_atlas_size;
    texture_atlas_size += w * h;
    printf("attribs: (%d, %d, %d)\n", cpu_textureData[i].w, cpu_textureData[i].h, cpu_textureData[i].s);
  }
  cpu_textureAtlas = new cl_float4[texture_atlas_size];
  int alloc = 0;
  for (int i = 0; i < texture_amt; i++)
  {
    int w, h, n;
    unsigned char *data = stbi_load(("res/textures/" + textures[i]).c_str(), &w, &h, &n, 0);
    for (int j = 0; j < w * h; j++)
      cpu_textureAtlas[j + alloc] = (cl_float4){{float(data[n * j]) / 255.0f, n <= 1 ? 0.0f : float(data[n * j + 1]) / 255.0f, n <= 2 ? 0.0f : float(data[n * j + 2]) / 255.0f, n <= 3 ? 0.0f : float(data[n * j + 3]) / 255.0f}};
    alloc += w * h;
    printf("s: %s, n: %d\n", textures[i].c_str(), n);
  }
}

void createBufferValues()
{

  // Assemble Useful Numbers
  cpu_usefulnums = new cl_uint[11];
  cpu_usefulnums[0] = (cl_uint)window_width;
  cpu_usefulnums[1] = (cl_uint)window_height;

  // Generate random seeds
  cpu_randoms = new cl_uint[window_width * window_height];
  for (int i = 0; i < window_width * window_height; i++)
  {
    cpu_randoms[i] = (cl_uint)(distribution(generator));
  }

  // Construct the scene
  buildScene();

  // IBL Loading
  string ibl_src_str = scn.iblPath;
  std::cout << ibl_src_str << endl;
  const char *ibl_src = ibl_src_str.c_str();

  HDRLoaderResult result;
  bool ret = HDRLoader::load(ibl_src, result);
  ibl_width = result.width;
  ibl_height = result.height;
  cpu_ibl = new cl_float4[ibl_width * ibl_height];
  float r, g, b;
  for (int i = 0; i < ibl_width * ibl_height; i++)
  {
    r = result.cols[3 * i];
    g = result.cols[3 * i + 1];
    b = result.cols[3 * i + 2];
    cpu_ibl[i] = (cl_float4){{r, g, b}};
  }
  cpu_usefulnums[2] = (cl_uint)ibl_width;
  cpu_usefulnums[3] = (cl_uint)ibl_height;

  cpu_usefulnums[4] = (cl_uint)sphere_amt;
  cpu_usefulnums[5] = (cl_uint)triangle_amt;
  cpu_usefulnums[6] = (cl_uint)bvhnode_amt;
  cpu_usefulnums[7] = (cl_uint)material_amt;
  cpu_usefulnums[8] = (cl_uint)medium_amt;
  cpu_usefulnums[9] = (cl_uint)samplesPerRun;

  bools = 0;
  bools |= scn.use_DOF;
  bools |= scn.use_IbL << 1;
  bools |= scn.use_ground << 2;

  cpu_usefulnums[10] = bools;

  initCamera();

  cpu_camera = new Camera();
  interactiveCamera->buildRenderCamera(cpu_camera);

  voidcolor = (cl_float3){{scn.background_color.x, scn.background_color.y, scn.background_color.z}};

  if (wavefront)
  {
    cpu_finished = new cl_uchar[window_width * window_height];
    cpu_actualIDs = new cl_int[window_width * window_height];
    cpu_chunks = new Chunk[2048];
  }
}

void createBufferValues_offline()
{

  // Assemble Useful Numbers
  cpu_usefulnums = new cl_uint[11];
  cpu_usefulnums[0] = (cl_uint)window_width;
  cpu_usefulnums[1] = (cl_uint)window_height;

  // Generate random seeds
  cpu_randoms = new cl_uint[window_width * window_height];
  for (int i = 0; i < window_width * window_height; i++)
  {
    cpu_randoms[i] = (cl_uint)(distribution(generator));
  }

  // Construct the scene
  buildScene();

  // IBL Loading
  string ibl_src_str = scn.iblPath;
  std::cout << ibl_src_str << endl;
  const char *ibl_src = ibl_src_str.c_str();

  HDRLoaderResult result;
  bool ret = HDRLoader::load(ibl_src, result);
  ibl_width = result.width;
  ibl_height = result.height;
  cpu_ibl = new cl_float4[ibl_width * ibl_height];
  float r, g, b;
  for (int i = 0; i < ibl_width * ibl_height; i++)
  {
    r = result.cols[3 * i];
    g = result.cols[3 * i + 1];
    b = result.cols[3 * i + 2];
    cpu_ibl[i] = (cl_float4){{r, g, b}};
  }
  cpu_usefulnums[2] = (cl_uint)ibl_width;
  cpu_usefulnums[3] = (cl_uint)ibl_height;

  cpu_usefulnums[4] = (cl_uint)sphere_amt;
  cpu_usefulnums[5] = (cl_uint)triangle_amt;
  cpu_usefulnums[6] = (cl_uint)bvhnode_amt;
  cpu_usefulnums[7] = (cl_uint)material_amt;
  cpu_usefulnums[8] = (cl_uint)medium_amt;
  cpu_usefulnums[9] = (cl_uint)samplesPerRun;

  bools = 0;
  bools |= scn.use_DOF;
  bools |= scn.use_IbL << 1;
  bools |= scn.use_ground << 2;

  cpu_usefulnums[10] = bools;

  initCamera();

  cpu_camera = new Camera();
  interactiveCamera->buildRenderCamera(cpu_camera);

  voidcolor = (cl_float3){{scn.background_color.x, scn.background_color.y, scn.background_color.z}};
}

void writeBufferValues()
{

  if (wavefront)
  {
    // Create point buffer on the OpenCL device
    cl_points = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));

    std::cout << "Created point buffer \n";

    // Create normal buffer on the OpenCL device
    cl_normals = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));

    std::cout << "Created normal buffer \n";

    // Create throughput buffer on the OpenCL device
    cl_throughputs = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));

    std::cout << "Created throughput buffer \n";

    // Create actual ID buffer on the OpenCL device
    cl_actualIDs = Buffer(context, CL_MEM_READ_WRITE, window_width * window_height * sizeof(cl_int));

    std::cout << "Created actual ID buffer \n";

    // Create finished buffer on the OpenCL device
    cl_finished = Buffer(context, CL_MEM_READ_WRITE, window_width * window_height * sizeof(cl_uchar));

    std::cout << "Created finished buffer \n";

    // Create material index buffer on the OpenCL device
    cl_mtlidxs = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_int));

    std::cout << "Created material index buffer \n";

    // Create ray buffer on the OpenCL device
    cl_rays = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * 3 * sizeof(cl_float3));

    std::cout << "Created ray buffer \n";

    // Create ray buffer on the OpenCL device
    cl_chunks = Buffer(context, CL_MEM_WRITE_ONLY, 2048 * sizeof(Chunk));

    std::cout << "Created chunk buffer \n";
  }

  // Create useful nums buffer on the OpenCL device
  cl_usefulnums = Buffer(context, CL_MEM_READ_ONLY, 11 * sizeof(cl_uint));
  queue.enqueueWriteBuffer(cl_usefulnums, CL_TRUE, 0, 11 * sizeof(cl_uint), cpu_usefulnums);

  std::cout << "Wrote useful numbers \n";

  // Create random buffer on the OpenCL device
  cl_randoms = Buffer(context, CL_MEM_READ_WRITE, window_width * window_height * sizeof(cl_uint));
  queue.enqueueWriteBuffer(cl_randoms, CL_TRUE, 0, window_width * window_height * sizeof(cl_uint), cpu_randoms);

  std::cout << "Wrote randoms \n";

  // Create ibl buffer on the OpenCL device
  cl_ibl = Buffer(context, CL_MEM_READ_ONLY, ibl_width * ibl_height * sizeof(cl_float4));
  queue.enqueueWriteBuffer(cl_ibl, CL_TRUE, 0, ibl_width * ibl_height * sizeof(cl_float4), cpu_ibl);

  std::cout << "Wrote IbL \n";

  // Create texture data buffer on the OpenCL device
  cl_textureData = Buffer(context, CL_MEM_READ_ONLY, texture_amt * sizeof(TextureData));
  queue.enqueueWriteBuffer(cl_textureData, CL_TRUE, 0, texture_amt * sizeof(TextureData), cpu_textureData);

  std::cout << "Wrote Texture Data \n";

  // Create ibl buffer on the OpenCL device
  cl_textureAtlas = Buffer(context, CL_MEM_READ_ONLY, texture_atlas_size * sizeof(cl_float4));
  queue.enqueueWriteBuffer(cl_textureAtlas, CL_TRUE, 0, texture_atlas_size * sizeof(cl_float4), cpu_textureAtlas);

  std::cout << "Wrote Texture Atlas \n";

  // Create sphere buffer on the OpenCL device
  cl_spheres = Buffer(context, CL_MEM_READ_ONLY, sphere_amt * sizeof(Sphere));
  queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_amt * sizeof(Sphere), cpu_spheres);

  std::cout << "Wrote spheres \n";

  // Create triangle buffer on the OpenCL device
  cl_triangles = Buffer(context, CL_MEM_READ_ONLY, triangle_amt * sizeof(Triangle));
  queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, triangle_amt * sizeof(Triangle), cpu_triangles);

  std::cout << "Wrote triangles \n";

  // Create triangle data buffer on the OpenCL device
  cl_triangleData = Buffer(context, CL_MEM_READ_ONLY, triangle_amt * sizeof(TriangleData));
  queue.enqueueWriteBuffer(cl_triangleData, CL_TRUE, 0, triangle_amt * sizeof(TriangleData), cpu_triangleData);

  std::cout << "Wrote triangle data \n";

  // Create BVH node buffer on the OpenCL device
  cl_nodes = Buffer(context, CL_MEM_READ_ONLY, bvhnode_amt * sizeof(BVHNode));
  queue.enqueueWriteBuffer(cl_nodes, CL_TRUE, 0, bvhnode_amt * sizeof(BVHNode), cpu_bvhs);

  std::cout << "Wrote BVH \n";

  // Create material buffer on the OpenCL device
  cl_materials = Buffer(context, CL_MEM_READ_ONLY, material_amt * sizeof(Material));
  queue.enqueueWriteBuffer(cl_materials, CL_TRUE, 0, material_amt * sizeof(Material), cpu_materials);

  std::cout << "Wrote materials \n";

  // Create medium buffer on the OpenCL device
  cl_mediums = Buffer(context, CL_MEM_READ_ONLY, medium_amt * sizeof(Medium));
  queue.enqueueWriteBuffer(cl_mediums, CL_TRUE, 0, medium_amt * sizeof(Medium), cpu_mediums);

  std::cout << "Wrote mediums \n";

  // Create camera buffer on the OpenCL device
  cl_camera = Buffer(context, CL_MEM_READ_ONLY, sizeof(Camera));
  queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(Camera), cpu_camera);

  std::cout << "Wrote camera \n";

  // create OpenCL buffer from OpenGL vertex buffer object
  cl_vbo = BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
  cl_vbos.push_back(cl_vbo);

  std::cout << "Wrote VBO \n";

  // reserve memory buffer on OpenCL device to hold image buffer for accumulated samples
  cl_accumbuffer = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));

  std::cout << "Created accumbuffer \n";
}

void writeBufferValues_offline()
{

  // Create useful nums buffer on the OpenCL device
  cl_usefulnums = Buffer(context, CL_MEM_READ_ONLY, 11 * sizeof(cl_uint));
  queue.enqueueWriteBuffer(cl_usefulnums, CL_TRUE, 0, 11 * sizeof(cl_uint), cpu_usefulnums);

  std::cout << "Wrote useful numbers \n";

  // Create random buffer on the OpenCL device
  cl_randoms = Buffer(context, CL_MEM_READ_WRITE, window_width * window_height * sizeof(cl_uint));
  queue.enqueueWriteBuffer(cl_randoms, CL_TRUE, 0, window_width * window_height * sizeof(cl_uint), cpu_randoms);

  std::cout << "Wrote randoms \n";

  // Create ibl buffer on the OpenCL device
  cl_ibl = Buffer(context, CL_MEM_READ_ONLY, ibl_width * ibl_height * sizeof(cl_float4));
  queue.enqueueWriteBuffer(cl_ibl, CL_TRUE, 0, ibl_width * ibl_height * sizeof(cl_float4), cpu_ibl);

  std::cout << "Wrote IbL \n";

  // Create texture data buffer on the OpenCL device
  cl_textureData = Buffer(context, CL_MEM_READ_ONLY, texture_amt * sizeof(TextureData));
  queue.enqueueWriteBuffer(cl_textureData, CL_TRUE, 0, texture_amt * sizeof(TextureData), cpu_textureData);

  std::cout << "Wrote Texture Data \n";

  // Create ibl buffer on the OpenCL device
  cl_textureAtlas = Buffer(context, CL_MEM_READ_ONLY, texture_atlas_size * sizeof(cl_float4));
  queue.enqueueWriteBuffer(cl_textureAtlas, CL_TRUE, 0, texture_atlas_size * sizeof(cl_float4), cpu_textureAtlas);

  std::cout << "Wrote Texture Atlas \n";

  // Create sphere buffer on the OpenCL device
  cl_spheres = Buffer(context, CL_MEM_READ_ONLY, sphere_amt * sizeof(Sphere));
  queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_amt * sizeof(Sphere), cpu_spheres);

  std::cout << "Wrote spheres \n";

  // Create triangle buffer on the OpenCL device
  cl_triangles = Buffer(context, CL_MEM_READ_ONLY, triangle_amt * sizeof(Triangle));
  queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, triangle_amt * sizeof(Triangle), cpu_triangles);

  std::cout << "Wrote triangles \n";

  // Create triangle buffer on the OpenCL device
  cl_triangleData = Buffer(context, CL_MEM_READ_ONLY, triangle_amt * sizeof(TriangleData));
  queue.enqueueWriteBuffer(cl_triangleData, CL_TRUE, 0, triangle_amt * sizeof(TriangleData), cpu_triangleData);

  std::cout << "Wrote triangle data \n";

  // Create BVH node buffer on the OpenCL device
  cl_nodes = Buffer(context, CL_MEM_READ_ONLY, bvhnode_amt * sizeof(BVHNode));
  queue.enqueueWriteBuffer(cl_nodes, CL_TRUE, 0, bvhnode_amt * sizeof(BVHNode), cpu_bvhs);

  std::cout << "Wrote BVH \n";

  // Create material buffer on the OpenCL device
  cl_materials = Buffer(context, CL_MEM_READ_ONLY, material_amt * sizeof(Material));
  queue.enqueueWriteBuffer(cl_materials, CL_TRUE, 0, material_amt * sizeof(Material), cpu_materials);

  std::cout << "Wrote materials \n";

  // Create medium buffer on the OpenCL device
  cl_mediums = Buffer(context, CL_MEM_READ_ONLY, medium_amt * sizeof(Medium));
  queue.enqueueWriteBuffer(cl_mediums, CL_TRUE, 0, medium_amt * sizeof(Medium), cpu_mediums);

  std::cout << "Wrote mediums \n";

  // Create camera buffer on the OpenCL device
  cl_camera = Buffer(context, CL_MEM_READ_ONLY, sizeof(Camera));
  queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(Camera), cpu_camera);

  std::cout << "Wrote camera \n";

  cl_output = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));

  // reserve memory buffer on OpenCL device to hold image buffer for accumulated samples
  cl_accumbuffer = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));

  std::cout << "Created accumbuffer \n";
}

void initCLKernel()
{

  // Create the kernel
  kernel = Kernel(program, "render_kernel");

  // specify kernel arguments
  kernel.setArg(0, cl_accumbuffer);
  kernel.setArg(1, cl_usefulnums);
  kernel.setArg(2, cl_randoms);
  kernel.setArg(3, cl_ibl);
  kernel.setArg(4, cl_vbo);
  kernel.setArg(5, cl_spheres);
  kernel.setArg(6, cl_triangles);
  kernel.setArg(7, cl_nodes);
  kernel.setArg(8, cl_materials);
  kernel.setArg(9, cl_mediums);
  kernel.setArg(10, cl_textureData);
  kernel.setArg(11, cl_textureAtlas);
  kernel.setArg(12, cl_triangleData);
  kernel.setArg(13, voidcolor);
  kernel.setArg(14, cl_camera);
  kernel.setArg(15, framenumber);
}

void initCLKernel_offline()
{

  // Create the kernel
  kernel = Kernel(program, "render_kernel");

  // specify kernel arguments
  kernel.setArg(0, cl_accumbuffer);
  kernel.setArg(1, cl_usefulnums);
  kernel.setArg(2, cl_randoms);
  kernel.setArg(3, cl_ibl);
  kernel.setArg(4, cl_spheres);
  kernel.setArg(5, cl_triangles);
  kernel.setArg(6, cl_nodes);
  kernel.setArg(7, cl_materials);
  kernel.setArg(8, cl_mediums);
  kernel.setArg(9, cl_textureData);
  kernel.setArg(10, cl_textureAtlas);
  kernel.setArg(11, cl_triangleData);
  kernel.setArg(12, voidcolor);
  kernel.setArg(13, cl_camera);

  final_kernel = Kernel(program, "final_kernel");

  final_kernel.setArg(0, cl_output);
  final_kernel.setArg(1, cl_accumbuffer);
  final_kernel.setArg(2, window_width);
  final_kernel.setArg(3, window_height);
  final_kernel.setArg(4, framenumber);
}

void initInitKernel()
{

  // Create the init kernel
  init_kernel = Kernel(program, "init_kernel");

  // specify init kernel arguments
  init_kernel.setArg(0, cl_rays);
  init_kernel.setArg(1, cl_throughputs);
  init_kernel.setArg(2, cl_actualIDs);
  init_kernel.setArg(3, cl_randoms);
  init_kernel.setArg(4, cl_camera);
  init_kernel.setArg(5, window_width);
  init_kernel.setArg(6, window_height);
  init_kernel.setArg(7, bools);
}

void initIntersectionKernels()
{

  // Create the intersection kernels
  intersection_kernel = Kernel(program, "intersection_kernel");
  intersectionfp_kernel = Kernel(program, "intersectionfp_kernel");

  // specify intersection kernel arguments
  intersection_kernel.setArg(0, cl_finished);
  intersection_kernel.setArg(1, cl_points);
  intersection_kernel.setArg(2, cl_normals);
  intersection_kernel.setArg(3, cl_mtlidxs);
  intersection_kernel.setArg(4, cl_rays);
  intersection_kernel.setArg(5, cl_spheres);
  intersection_kernel.setArg(6, cl_triangles);
  intersection_kernel.setArg(7, cl_nodes);
  intersection_kernel.setArg(8, cl_actualIDs);
  intersection_kernel.setArg(9, sphere_amt);
  intersection_kernel.setArg(10, bvhnode_amt);
  intersection_kernel.setArg(11, bools);

  // specify intersection first pass kernel arguments
  intersectionfp_kernel.setArg(0, cl_finished);
  intersectionfp_kernel.setArg(1, cl_points);
  intersectionfp_kernel.setArg(2, cl_normals);
  intersectionfp_kernel.setArg(3, cl_mtlidxs);
  intersectionfp_kernel.setArg(4, cl_rays);
  intersectionfp_kernel.setArg(5, cl_spheres);
  intersectionfp_kernel.setArg(6, cl_triangles);
  intersectionfp_kernel.setArg(7, cl_nodes);
  intersectionfp_kernel.setArg(8, sphere_amt);
  intersectionfp_kernel.setArg(9, bvhnode_amt);
  intersectionfp_kernel.setArg(10, bools);
}

void initShadingKernels()
{

  // Create the shading kernels
  shading_kernel = Kernel(program, "shading_kernel");
  shadingfp_kernel = Kernel(program, "shadingfp_kernel");

  // specify shading kernel arguments
  shading_kernel.setArg(0, cl_rays);
  shading_kernel.setArg(1, cl_finished);
  shading_kernel.setArg(2, cl_globalworkgroupsize);
  shading_kernel.setArg(3, cl_accumbuffer);
  shading_kernel.setArg(4, cl_throughputs);
  shading_kernel.setArg(5, cl_mtlidxs);
  shading_kernel.setArg(6, cl_points);
  shading_kernel.setArg(7, cl_normals);
  shading_kernel.setArg(8, cl_materials);
  shading_kernel.setArg(9, cl_ibl);
  shading_kernel.setArg(10, cl_actualIDs);
  shading_kernel.setArg(11, cl_randoms);
  shading_kernel.setArg(12, ibl_width);
  shading_kernel.setArg(13, ibl_height);
  shading_kernel.setArg(14, voidcolor);
  shading_kernel.setArg(15, bools);
  shading_kernel.setArg(16, 0);

  // specify shading first pass kernel arguments
  shadingfp_kernel.setArg(0, cl_rays);
  shadingfp_kernel.setArg(1, cl_finished);
  shadingfp_kernel.setArg(2, cl_globalworkgroupsize);
  shadingfp_kernel.setArg(3, cl_accumbuffer);
  shadingfp_kernel.setArg(4, cl_throughputs);
  shadingfp_kernel.setArg(5, cl_mtlidxs);
  shadingfp_kernel.setArg(6, cl_points);
  shadingfp_kernel.setArg(7, cl_normals);
  shadingfp_kernel.setArg(8, cl_materials);
  shadingfp_kernel.setArg(9, cl_ibl);
  shadingfp_kernel.setArg(10, cl_randoms);
  shadingfp_kernel.setArg(11, ibl_width);
  shadingfp_kernel.setArg(12, ibl_height);
  shadingfp_kernel.setArg(13, voidcolor);
  shadingfp_kernel.setArg(14, bools);
  shadingfp_kernel.setArg(15, 0);
}

void initReassignKernels()
{

  // Create a kernel (entry point in the OpenCL source program)
  reassign_kernel = Kernel(program, "reassign_kernel");
  reassignfp_kernel = Kernel(program, "reassignfp_kernel");

  // specify OpenCL kernel arguments
  reassign_kernel.setArg(0, cl_actualIDs);
  reassign_kernel.setArg(1, cl_finished);
  reassign_kernel.setArg(2, cl_chunks);
  reassign_kernel.setArg(3, 0);
  reassign_kernel.setArg(4, 0);

  reassignfp_kernel.setArg(0, cl_actualIDs);
  reassignfp_kernel.setArg(1, cl_finished);
  reassignfp_kernel.setArg(2, cl_chunks);
  reassignfp_kernel.setArg(3, 0);
  reassignfp_kernel.setArg(4, 0);
}

void initShiftKernels()
{

  // Create a kernel (entry point in the OpenCL source program)
  shift_kernel = Kernel(program, "shift_kernel");
  shift_kernel_alt = Kernel(program, "shift_kernel");

  // specify OpenCL kernel arguments
  shift_kernel.setArg(0, cl_actualIDs);
  shift_kernel.setArg(1, cl_actualIDs_new);
  shift_kernel.setArg(2, cl_chunks);
  shift_kernel.setArg(3, cl_chunks_new);

  shift_kernel_alt.setArg(0, cl_actualIDs_new);
  shift_kernel_alt.setArg(1, cl_actualIDs);
  shift_kernel_alt.setArg(2, cl_chunks_new);
  shift_kernel_alt.setArg(3, cl_chunks);
}

void initFinalKernel()
{

  // Create a kernel (entry point in the OpenCL source program)
  final_kernel = Kernel(program, "final_kernel");

  // specify OpenCL kernel arguments
  final_kernel.setArg(0, cl_vbo);
  final_kernel.setArg(1, cl_accumbuffer);
  final_kernel.setArg(2, window_width);
  final_kernel.setArg(3, window_height);
  final_kernel.setArg(4, framenumber);
}

// TODO wavefront-relevant
void initCLKernels()
{

  initInitKernel();

  std::cout << "Initialized Init Kernel\n";

  initIntersectionKernels();

  std::cout << "Initialized Intersection Kernels\n";

  initShadingKernels();

  std::cout << "Initialized Shading Kernels\n";

  initReassignKernels();

  std::cout << "Initialized Reassign Kernels\n";

  initFinalKernel();

  std::cout << "Initialized Final Kernel\n";
}

// TODO wavefront-relevant
void runKernels()
{

  std::size_t cl_int_size = sizeof(cl_int);

  std::size_t global_work_size = window_width * window_height;
  std::size_t global_work_size_tmp = window_width * window_height;
  std::size_t local_work_size = init_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

  // Ensure the global work size is a multiple of local work size
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

  // Make sure OpenGL is done using the VBOs
  glFinish();

  // Pass in the vector of VBO buffer objects
  queue.enqueueAcquireGLObjects(&cl_vbos);
  queue.finish();

  // std::chrono::duration<double> elapsedinit;
  // std::chrono::duration<double> elapsedintersect;
  // std::chrono::duration<double> elapsedshading;
  // std::chrono::duration<double> elapsedreassign;
  // std::chrono::duration<double> elapsedfinish;

  // // Start timer
  // auto start = std::chrono::high_resolution_clock::now();

  // Launch init kernel
  queue.enqueueNDRangeKernel(init_kernel, NULL, global_work_size, local_work_size);
  queue.flush();
  // queue.finish();

  // // End timer
  // auto finish = std::chrono::high_resolution_clock::now();

  // elapsedinit += finish - start;

  int bounces = 1500;

  for (int n = 0; n < bounces && global_work_size > 0; n++)
  {

    bool firstpass = (n == 0 || global_work_size == window_width * window_height);

    // // Start timer
    // start = std::chrono::high_resolution_clock::now();

    shadingfp_kernel.setArg(15, n);
    shading_kernel.setArg(16, n);

    if (firstpass)
    {
      // Launch intersection kernel
      queue.enqueueNDRangeKernel(intersectionfp_kernel, NULL, global_work_size, local_work_size);
      queue.flush();
      // queue.finish();

      // End timer
      // finish = std::chrono::high_resolution_clock::now();

      // elapsedintersect += finish - start;

      // Start timer
      // start = std::chrono::high_resolution_clock::now();

      // Launch shading kernel
      queue.enqueueNDRangeKernel(shadingfp_kernel, NULL, global_work_size, local_work_size);
      queue.flush();
    }
    else
    {
      // Launch intersection kernel
      queue.enqueueNDRangeKernel(intersection_kernel, NULL, global_work_size, local_work_size);
      queue.flush();
      // queue.finish();

      // End timer
      // finish = std::chrono::high_resolution_clock::now();

      // elapsedintersect += finish - start;

      shading_kernel.setArg(16, n);

      // Start timer
      // start = std::chrono::high_resolution_clock::now();

      // Launch shading kernel
      queue.enqueueNDRangeKernel(shading_kernel, NULL, global_work_size, local_work_size);
      queue.flush();
    }
    // queue.finish();

    // // End timer
    // finish = std::chrono::high_resolution_clock::now();

    // elapsedshading += finish - start;

    // std::chrono::duration<double> elapsed = finish - start;

    // printf("Shaded %d pixels in %f s.\n", global_work_size, elapsed.count());

    /*
         Calculate reassignment if there are more iterations for the current frame
         */
    if (n < bounces - 1)
    {

      // // Start timer
      // start = std::chrono::high_resolution_clock::now();

      global_work_size = global_work_size_tmp;

      int total_size = global_work_size;

      if (total_size >= 1024 * 16)
      {

        // // Start timer
        // auto start2 = std::chrono::high_resolution_clock::now();

        std::size_t local_thread_count = 1;

        int thread_count = 1024;
        int unit_size = (int)ceil(((float)total_size) / ((float)thread_count));
        // if(total_size % thread_count != 0)
        //     unit_size += 1;

        int chunk_amt = (int)ceil(((float)total_size) / ((float)unit_size));

        std::size_t reassign_work_size = chunk_amt;

        // printf("Total size: %d, Unit size: %d.\n", total_size, unit_size);

        if (reassign_work_size % local_thread_count != 0)
          reassign_work_size = (reassign_work_size / local_thread_count + 1) * local_thread_count;

        // auto start2 = std::chrono::high_resolution_clock::now();

        if (firstpass)
        {
          reassignfp_kernel.setArg(3, total_size);
          reassignfp_kernel.setArg(4, unit_size);
          queue.enqueueNDRangeKernel(reassignfp_kernel, NULL, reassign_work_size, local_thread_count);
        }
        else
        {
          reassign_kernel.setArg(3, total_size);
          reassign_kernel.setArg(4, unit_size);
          queue.enqueueNDRangeKernel(reassign_kernel, NULL, reassign_work_size, local_thread_count);
        }
        queue.flush();
        queue.finish();
        // // End timer
        // auto finish2 = std::chrono::high_resolution_clock::now();

        // std::chrono::duration<double> elapsed = finish2 - start2;

        // printf("Computed %d chunks in %f s.\n", chunk_amt, elapsed.count());

        // start2 = std::chrono::high_resolution_clock::now();
        queue.enqueueReadBuffer(cl_chunks, CL_TRUE, 0, chunk_amt * sizeof(Chunk), cpu_chunks);

        Shift *shifts = new Shift[chunk_amt - 1];
        int offset = 0;
        int nzshifts = 0; //-1;
        Chunk next_chunk;
        queue.finish();
        Chunk curr_chunk = cpu_chunks[0];
        // int readfrom = 0;
        // int writeto = 0;
        for (int i = 1; i < chunk_amt; i++)
        {
          next_chunk = cpu_chunks[i];
          int moving = next_chunk.i - curr_chunk.f;
          if (moving > 0)
          {
            // if(nzshifts < 0) {
            //     readfrom = curr_chunk.f;
            //     queue.enqueueReadBuffer(cl_actualIDs, CL_FALSE, readfrom * cl_int_size, (global_work_size - readfrom) * cl_int_size, cpu_actualIDs);
            // }
            // else {
            //     shifts[nzshifts] = Shift(curr_chunk.i - offset - writeto, curr_chunk.f - readfrom, moving * cl_int_size);
            // }
            shifts[nzshifts] = Shift(curr_chunk.i - offset, curr_chunk.f, moving * cl_int_size);
            nzshifts++;
          }
          int size = curr_chunk.f - curr_chunk.i;
          // if(offset == 0 && size > 0) {
          //     writeto = curr_chunk.i;
          // }
          offset += size;
          curr_chunk = next_chunk;
        }
        offset += curr_chunk.f - curr_chunk.i;
        // queue.finish();

        if (offset > 0)
        {
          if (global_work_size == offset)
            break;
          queue.enqueueReadBuffer(cl_actualIDs, CL_TRUE, 0, global_work_size * cl_int_size, cpu_actualIDs);
          global_work_size -= offset;

          // cl_int *naids = new cl_int[global_work_size];
          // memcpy(naids, cpu_actualIDs, cpu_chunks[0].i * cl_int_size);
          // thread *threads = new thread[nzshifts];
          Shift s;
          queue.finish();
          for (int i = 0; i < nzshifts; i++)
          {
            s = shifts[i];
            memmove(cpu_actualIDs + s.dst, cpu_actualIDs + s.src, s.size);

            // threads[i] = thread(memcpy, naids + s.dst, cpu_actualIDs + s.src, s.size);
          }

          // for(int i = 0; i < nzshifts; i++) {
          //     if(shifts[i].size > 0)
          //         threads[i].join();
          // }

          // queue.enqueueFillBuffer(cl_finished, 0, 0, global_work_size * sizeof(cl_uchar));
          // queue.enqueueWriteBuffer(cl_actualIDs, CL_TRUE, writeto * cl_int_size, (global_work_size - writeto) * cl_int_size, cpu_actualIDs);
          queue.enqueueWriteBuffer(cl_actualIDs, CL_TRUE, cl_int_size, global_work_size * cl_int_size, cpu_actualIDs);
          queue.finish();
        }

        // finish2 = std::chrono::high_resolution_clock::now();

        // elapsed = finish2 - start2;

        // printf("Did the rest in %f s.\n", chunk_amt, elapsed.count());

        // // End timer
        // auto finish = std::chrono::high_resolution_clock::now();

        // std::chrono::duration<double> elapsed = finish - start;

        // printf("Killed %d threads and reindexed and wrote the remainder in %f s.\n", offset, elapsed.count());
      }
      else
      {

        queue.enqueueReadBuffer(cl_finished, CL_TRUE, 0, global_work_size * sizeof(cl_uchar), cpu_finished);
        if (!firstpass)
          queue.enqueueReadBuffer(cl_actualIDs, CL_TRUE, 0, global_work_size * cl_int_size, cpu_actualIDs);

        int global_work_size_old = global_work_size;

        int currentpos = 0;

        queue.finish();
        if (firstpass)
        {
          for (int i = 0; i < global_work_size_old; i++)
          {
            if (!cpu_finished[i])
            {
              cpu_actualIDs[currentpos++] = i;
            }
          }
        }
        else
        {
          for (int i = 0; i < global_work_size_old; i++)
          {
            if (!cpu_finished[i])
            {
              cpu_actualIDs[currentpos++] = cpu_actualIDs[i];
            }
          }
        }
        global_work_size = currentpos;

        if (global_work_size_old - global_work_size)
        {
          if (global_work_size == 0)
            break;
          queue.enqueueWriteBuffer(cl_actualIDs, CL_TRUE, 0, global_work_size * cl_int_size, cpu_actualIDs);
          queue.finish();
        }

        // Set global_work_size to kernel-computed value;
      }
      global_work_size_tmp = global_work_size;

      intersection_kernel.setArg(12, (int)global_work_size);
      shading_kernel.setArg(17, (int)global_work_size);
      // local_work_size = init_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

      // if(global_work_size < local_work_size * 16 && global_work_size >= 16)
      //     local_work_size = global_work_size / 16;
      // else if(global_work_size < local_work_size)
      //     local_work_size = 1;

      // Ensure the global work size is a multiple of local work size
      if (global_work_size % local_work_size != 0)
        global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

      // // End timer
      // finish = std::chrono::high_resolution_clock::now();

      // elapsedreassign += finish - start;
    }
  }

  global_work_size = window_width * window_height;
  // local_work_size = init_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

  while (global_work_size < 4 * local_work_size)
    local_work_size = (std::size_t)std::max(1, (int)local_work_size / 2);

  // Ensure the global work size is a multiple of local work size
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

  // // Start timer
  // start = std::chrono::high_resolution_clock::now();

  // Launch final kernel
  queue.enqueueNDRangeKernel(final_kernel, NULL, global_work_size, local_work_size);
  queue.finish();

  // // End timer
  // finish = std::chrono::high_resolution_clock::now();

  // elapsedfinish += finish - start;

  // Release the VBOs so OpenGL can play with them
  queue.enqueueReleaseGLObjects(&cl_vbos);
  queue.finish();

  // printf("Frame complete: \n - Initialize: %f s\n - Intersection/Shading: %f s\n - Reassign: %f s\n - Finalize: %f s\n",
  //        elapsedinit.count(),
  //        elapsedshading.count(),
  //        elapsedreassign.count(),
  //        elapsedfinish.count()
  //        );

  // printf("Frame complete: \n - Initialize: %f s\n - Intersection: %f s\n - Shading: %f s\n - Reassign: %f s\n - Finalize: %f s\n",
  //        elapsedinit.count(),
  //        elapsedintersect.count(),
  //        elapsedshading.count(),
  //        elapsedreassign.count(),
  //        elapsedfinish.count()
  //        );
}

void runKernel()
{
  // every pixel in the image has its own thread or "work item",
  // so the total amount of work items equals the number of pixels
  std::size_t global_work_size = window_width * window_height;
  std::size_t local_work_size = 64; // kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

  // Ensure the global work size is a multiple of local work size
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

  // Make sure OpenGL is done using the VBOs
  glFinish();

  // Pass in the vector of VBO buffer objects
  queue.enqueueAcquireGLObjects(&cl_vbos);
  queue.finish();

  auto start = std::chrono::high_resolution_clock::now();

  // Launch the kernel
  queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size);
  queue.finish();

  auto finish = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = finish - start;

  printf("Megakernel took %f seconds.\n", elapsed.count());

  // Release the VBOs so OpenGL can play with them
  queue.enqueueReleaseGLObjects(&cl_vbos);
  queue.finish();
}

void runKernel_offline()
{
  // every pixel in the image has its own thread or "work item",
  // so the total amount of work items equals the number of pixels
  std::size_t global_work_size = window_width * window_height;
  std::size_t local_work_size = 64; // kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

  // Ensure the global work size is a multiple of local work size
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

  int i = 0;

  queue.enqueueFillBuffer(cl_accumbuffer, 0, 0, window_width * window_height * sizeof(cl_float3));
  queue.finish();

  for (; i < samples; i++)
  {

    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel

    queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size);
    queue.finish();

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;

    printf("Megakernel took %f seconds.\n", elapsed.count());

    printf("%d/%d samples per pixel\n", i + 1, samples);

    if ((i + 1) % sample_interval == 0)
    {
      final_kernel.setArg(4, i + 1);

      global_work_size = window_width * window_height;
      // local_work_size = init_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

      while (global_work_size < 4 * local_work_size)
        local_work_size = (std::size_t)std::max(1, (int)local_work_size / 2);

      // Ensure the global work size is a multiple of local work size
      if (global_work_size % local_work_size != 0)
        global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

      // Launch final kernel
      queue.enqueueNDRangeKernel(final_kernel, NULL, global_work_size, local_work_size);
      queue.finish();

      queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, window_width * window_height * sizeof(cl_float3), cpu_output);
      queue.finish();

      saveImage();
    }
  }
}

void render()
{

  if (buffer_reset)
  {
    float arg = 0;
    queue.enqueueWriteBuffer(cl_randoms, CL_TRUE, 0, window_width * window_height * sizeof(cl_uint), cpu_randoms);
    queue.finish();
    queue.enqueueFillBuffer(cl_accumbuffer, arg, 0, window_width * window_height * sizeof(cl_float3));
    queue.finish();
    framenumber = 0;
  }
  if (norm_mode && !wavefront)
  {
    framenumber = -2;
  }

  buffer_reset = false;
  framenumber++;

  interactiveCamera->buildRenderCamera(cpu_camera);
  queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(Camera), cpu_camera);
  queue.finish();

  if (wavefront)
  {
    // cpu_actualIDs = new cl_int[window_width * window_height];
    // cpu_chunks = new Chunk[2048];
    // queue.enqueueFillBuffer(cl_actualIDs, CL_TRUE, 0, window_width * window_height * sizeof(cl_int));
    queue.enqueueFillBuffer(cl_finished, CL_TRUE, 0, window_width * window_height * sizeof(cl_uchar));
    queue.finish();
  }

  if (wavefront)
  {
    // init_kernel.setArg(4, cl_camera);
    final_kernel.setArg(4, framenumber - 1);
  }
  else
  {
    // kernel.setArg(11, cl_camera);
    kernel.setArg(15, framenumber - 1);
  }

  // std::cout << "Running kernels...\n";

  if (wavefront)
    runKernels();
  else
    runKernel();

  drawGL();

  std::cout << samplesPerRun * framenumber << "\n";
}

void render_offline()
{

  interactiveCamera->buildRenderCamera(cpu_camera);
  queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(Camera), cpu_camera);
  queue.finish();

  runKernel_offline();
}

void cleanUp()
{
  delete cpu_chunks;
  delete cpu_finished;
  delete cpu_actualIDs;
  // delete cpu_accumbuffer;
  delete cpu_randoms;
  // delete cpu_usefulnums;
  // delete cpu_ibl;
  // delete cpu_spheres;
  // delete cpu_triangles;
  // delete cpu_triangleData;
  // delete cpu_bvhs;
  // delete cpu_materials;
  // delete cpu_mediums;
  delete cpu_camera;
}

void cleanUpInit()
{
  delete cpu_usefulnums;
  delete cpu_ibl;
  delete cpu_spheres;
  delete cpu_triangles;
  delete cpu_triangleData;
  delete cpu_bvhs;
  delete cpu_materials;
  delete cpu_mediums;
}

void initCamera()
{
  delete interactiveCamera;

  Vec3 cam_pos = scn.cam_pos;
  Vec3 cam_fd = scn.cam_fd;
  Vec3 cam_up = scn.cam_up;
  float cam_focal_distance = scn.cam_focal_distance;
  float cam_aperture_radius = scn.cam_aperture_radius;

  interactiveCamera = new InteractiveCamera(cam_pos, cam_fd, cam_up, cam_focal_distance, cam_aperture_radius);
}

int main_offline(int arc, char **argv)
{

  // Load the config file
  loadConfig("config", &window_width, &window_height, &scn_path, &interactive);

  std::cout << "Configurations loaded \n";

  // Init OpenCL
  initOpenCL_offline();

  std::cout << "OpenCL initialized \n";

  cpu_output = new cl_float3[window_width * window_height];

  // Create Buffer Values
  createBufferValues_offline();

  std::cout << "Buffer values created \n";

  // Write Buffer Values
  writeBufferValues_offline();

  std::cout << "Buffer values written \n";

  // Clean up initialization
  cleanUpInit();

  std::cout << "Cleaned up init \n";

  // intitialize the kernel
  initCLKernel_offline();

  std::cout << "CL Kernel initialized \n";

  render_offline();

  // release memory
  cleanUp();

  return 0;
}

int main(int argc, char **argv)
{

  if (OFFLINE)
  {
    return main_offline(argc, argv);
  }

  // Load the config file
  loadConfig("config", &window_width, &window_height, &scn_path, &interactive);

  std::cout << "Configurations loaded \n";

  // Init OpenGL
  initGL(argc, argv);

  std::cout << "OpenGL initialized \n";

  // Init OpenCL
  initOpenCL();

  std::cout << "OpenCL initialized \n";

  // Create the display VBO
  createVBO(&vbo);

  std::cout << "VBO created \n";

  // Start the timer
  Timer(0);

  std::cout << "Timer started \n";

  glFinish();

  std::cout << "glFinish executed \n";

  // Create Buffer Values
  createBufferValues();

  std::cout << "Buffer values created \n";

  // Write Buffer Values
  writeBufferValues();

  std::cout << "Buffer values written \n";

  // Clean up initialization
  cleanUpInit();

  std::cout << "Cleaned up init \n";

  // intitialize the kernel
  if (wavefront)
    initCLKernels();
  else
    initCLKernel();

  std::cout << "CL Kernel initialized \n";

  // start rendering continuously
  glutMainLoop();

  std::cout << "glutMainLoop executed \n";

  // release memory
  cleanUp();

  system("PAUSE");

  return 0;
}
