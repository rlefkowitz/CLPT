#define PI 3.141592653589793238f

#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <vector>
#include "cl.hpp"
#include "cl_gl_interop.h"
#include "configloader.h"
#include "hdrloader/hdrloader.h"
#include "sceneloader.h"
#include "user_interaction.h"

using namespace std;
using namespace cl;

bool profiling = false;

const int samples = 2048;
const int samplesPerRun = 1;

// Image-based Lighting variables
int ibl_width;
int ibl_height;
cl_float4* cpu_ibl;

// Background color
cl_float3 voidcolor;

// Sphere variables
Sphere* cpu_spheres;
int sphere_amt;

// Triangle variables
Triangle* cpu_triangles;
int triangle_amt;

// BVHNode variables
BVHNode* cpu_bvhs;
int bvhnode_amt;

// Material variables
Material* cpu_materials;
int material_amt;

// Medium variables
Medium* cpu_mediums;
int medium_amt;

// Finished variable
cl_uchar* cpu_finished;

// Actual ID variable
cl_int* cpu_actualIDs;

// Scene variables
string scn_path;
Scene scn;

// store booleans
unsigned char bools;

// Random generator
default_random_engine generator;
uniform_int_distribution<int> distribution(1, (1 << 31) - 2);

// Camera variable
Camera* cpu_camera = NULL;

// Host (CPU) cl_uint buffer variables
cl_uint* cpu_randoms;
cl_uint* cpu_usefulnums;

// OpenCL boiler plate variables
Device device;
CommandQueue queue;
Kernel kernel;
Kernel init_kernel;
Kernel intersection_kernel;
Kernel shading_kernel;
Kernel rmo_kernel;
Kernel rmf_kernel;
Kernel final_kernel;
Context context;
Program program;

// Device (CPU/GPU) Buffers
Buffer cl_usefulnums;
Buffer cl_mediums;

cl_uint *global_work_group_size;
Buffer cl_globalworkgroupsize;
Buffer cl_rays;
Buffer cl_randoms;
Buffer cl_camera;
Buffer cl_actualIDs;
Buffer cl_actualIDsTemp;
Buffer cl_finished;
Buffer cl_points;
Buffer cl_normals;
Buffer cl_mtlidxs;
Buffer cl_spheres;
Buffer cl_triangles;
Buffer cl_nodes;
Buffer cl_accumbuffer;
Buffer cl_throughputs;
Buffer cl_materials;
Buffer cl_ibl;
BufferGL cl_vbo;
vector<Memory> cl_vbos;

unsigned int framenumber = 0;

void pickPlatform(Platform& platform, const vector<Platform>& platforms){
    
    if (platforms.size() == 1) platform = platforms[0];
    else{
        int input = 0;
        cout << "\nChoose an OpenCL platform: ";
        cin >> input;
        
        // handle incorrect user input
        while (input < 1 || input > platforms.size()){
            cin.clear(); //clear errors/bad flags on cin
            cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
            cout << "No such option. Choose an OpenCL platform: ";
            cin >> input;
        }
        platform = platforms[input - 1];
    }
}

void pickDevice(Device& device, const vector<Device>& devices){
    
    if (devices.size() == 1) device = devices[0];
    else{
        int input = 0;
        cout << "\nChoose an OpenCL device: ";
        cin >> input;
        
        // handle incorrect user input
        while (input < 1 || input > devices.size()){
            cin.clear(); //clear errors/bad flags on cin
            cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
            cout << "No such option. Choose an OpenCL device: ";
            cin >> input;
        }
        device = devices[input - 1];
    }
}

void printErrorLog(const Program& program, const Device& device) {
    
    // Get the error log and print to console
    string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    cerr << "Build log:" << std::endl << buildlog << std::endl;
    
    // Print the error log to a file
    FILE *log = fopen("errorlog.txt", "w");
    fprintf(log, "%s\n", buildlog.c_str());
    cout << "Error log saved in 'errorlog.txt'" << endl;
    system("PAUSE");
    exit(1);
}

void initOpenCL()
{
    // Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
    vector<Platform> platforms;
    Platform::get(&platforms);
    cout << "Available OpenCL platforms : " << endl << endl;
    for (int i = 0; i < platforms.size(); i++)
    cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
    
    // Pick one platform
    Platform platform;
    pickPlatform(platform, platforms);
    cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;
    
    // Get available OpenCL devices on platform
    vector<Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    
    cout << "Available OpenCL devices on this platform: " << endl << endl;
    for (int i = 0; i < devices.size(); i++){
        cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
        cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
        cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl << endl;
    }
    
    // Pick one device
    pickDevice(device, devices);
    cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl;
    cout << "\t\t\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
    cout << "\t\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
    
    // Create an OpenCL context on that device.
    // Windows specific OpenCL-OpenGL interop
    CGLContextObj     kCGLContext     = CGLGetCurrentContext();
    CGLShareGroupObj  kCGLShareGroup  = CGLGetShareGroup(kCGLContext);
    
    cl_context_properties properties[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties) kCGLShareGroup,
        0
    };
    
    // Create a command queue
    context = Context(device, properties);
    cout << "Created context\n";
    if(profiling) {
        queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
    } else {
        queue = CommandQueue(context, device);
    }
    cout << "Created queue\n";
    
    // Convert the OpenCL source code to a string// Convert the OpenCL source code to a string
    string source;
    ifstream file("opencl_kernel.cl");
    if (!file){
        cout << "\nNo OpenCL file found!" << endl << "Exiting..." << endl;
        system("PAUSE");
        exit(1);
    }
    while (!file.eof()){
        char line[256];
        file.getline(line, 255);
        source += line;
    }
    
    const char* kernel_source = source.c_str();
    
    // Create an OpenCL program with source
    program = Program(context, kernel_source);
    cout << "Created program\n";
    
    // Build the program for the selected device
    cl_int result = program.build({ device }); // "-cl-fast-relaxed-math"
    cout << "Built program on device\n";
    if (result) cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
    if (result == CL_BUILD_PROGRAM_FAILURE) printErrorLog(program, device);
    
}

void buildScene() {
    
    if(!loadScene(scn, scn_path)) {
        printf("Error: could not load scene!");
        exit(1);
    }
    
    vector<Material> materials = scn.materials;
    vector<Medium> mediums = scn.mediums;
    vector<Sphere> spheres = scn.spheres;
    vector<Triangle> triangles = scn.triangles;
    vector<BVHNode> nodes = scn.nodes;

    /*
     Construct the BVH for all triangles added to the scene, if any.
     */
    if(triangles.size() > 0) {
        printf("Building BVH for scene with %d triangles...\n", triangles.size());
        nodes = build(triangles);
        printf("BVH finished for scene with %d triangles!\n\n", triangles.size());
    }
    
    
    sphere_amt = spheres.size();
    printf("Spheres: %d\n", sphere_amt);
    
    cpu_spheres = new Sphere[sphere_amt];
    for(int i = 0; i < sphere_amt; i++)
        cpu_spheres[i] = spheres[i];
    spheres.clear();
    
    
    triangle_amt = triangles.size();
    printf("Triangles: %d\n", triangle_amt);
    
    cpu_triangles = new Triangle[triangle_amt];
    for(int i = 0; i < triangle_amt; i++)
        cpu_triangles[i] = triangles[i];
    triangles.clear();
    
    
    bvhnode_amt = nodes.size();
    printf("BVH Nodes: %d\n", bvhnode_amt);
    
    cpu_bvhs = new BVHNode[bvhnode_amt];
    for(int i = 0; i < bvhnode_amt; i++)
        cpu_bvhs[i] = nodes[i];
    nodes.clear();
    
    
    material_amt = materials.size();
    printf("Materials: %d\n", material_amt);
    
    cpu_materials = new Material[material_amt];
    for(int i = 0; i < material_amt; i++)
        cpu_materials[i] = materials[i];
    materials.clear();
    
    
    medium_amt = mediums.size();
    printf("Mediums: %d\n", medium_amt);
    
    cpu_mediums = new Medium[medium_amt];
    for(int i = 0; i < medium_amt; i++)
        cpu_mediums[i] = mediums[i];
    mediums.clear();
    
}

void createBufferValues() {
    
    // Assemble Useful Numbers
    cpu_usefulnums = new cl_uint[11];
    cpu_usefulnums[0] = (cl_uint) window_width;
    cpu_usefulnums[1] = (cl_uint) window_height;
    
    // Generate random seeds
    cpu_randoms = new cl_uint[window_width * window_height];
    for(int i = 0; i < window_width * window_height; i++) {
        cpu_randoms[i] = (cl_uint) (distribution(generator));
    }
    
    // Construct the scene
    buildScene();
    
    // IBL Loading
    string ibl_src_str = scn.iblPath;
    cout << ibl_src_str << endl;
    const char* ibl_src = ibl_src_str.c_str();
    
    HDRLoaderResult result;
    bool ret = HDRLoader::load(ibl_src, result);
    ibl_width = result.width;
    ibl_height = result.height;
    cpu_ibl = new cl_float3[ibl_width * ibl_height];
    float r, g, b;
    for(int i = 0; i < ibl_width * ibl_height; i++) {
        r = result.cols[3*i];
        g = result.cols[3*i+1];
        b = result.cols[3*i+2];
        cpu_ibl[i] = (cl_float3) {{r, g, b}};
    }
    cpu_usefulnums[2] = (cl_uint) ibl_width;
    cpu_usefulnums[3] = (cl_uint) ibl_height;
    
    cpu_usefulnums[4] = (cl_uint) sphere_amt;
    cpu_usefulnums[5] = (cl_uint) triangle_amt;
    cpu_usefulnums[6] = (cl_uint) bvhnode_amt;
    cpu_usefulnums[7] = (cl_uint) material_amt;
    cpu_usefulnums[8] = (cl_uint) medium_amt;
    cpu_usefulnums[9] = (cl_uint) samplesPerRun;

    bools = 0;
    bools |= scn.use_DOF;
    bools |= scn.use_IbL << 1;
    bools |= scn.use_ground << 2;
    
    cpu_usefulnums[10] = bools;
    
    initCamera();
    
    cpu_camera = new Camera();
    interactiveCamera->buildRenderCamera(cpu_camera);
    
    voidcolor = (cl_float3) {{scn.background_color.x, scn.background_color.y, scn.background_color.z}};

    cpu_finished = new cl_uchar[window_width * window_height];
    cpu_actualIDs = new cl_int[window_width * window_height];
    
}

void writeBufferValues() {
    
    // Create point buffer on the OpenCL device
    cl_globalworkgroupsize = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_uint));
    global_work_group_size = (cl_uint *)queue.enqueueMapBuffer(cl_globalworkgroupsize, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_uint));  
    *global_work_group_size = 0;
    queue.enqueueUnmapMemObject(cl_globalworkgroupsize, global_work_group_size); 
    
    cout << "Created global work group size buffer \n";
    
    // Create point buffer on the OpenCL device
    cl_points = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));
    
    cout << "Created point buffer \n";
    
    // Create normal buffer on the OpenCL device
    cl_normals = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));
    
    cout << "Created normal buffer \n";
    
    // Create throughput buffer on the OpenCL device
    cl_throughputs = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));
    
    cout << "Created throughput buffer \n";
    
    // Create actual ID buffer on the OpenCL device
    cl_actualIDs = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_int));
    
    cout << "Created actual ID buffer \n";
    
    // Create actual ID temp buffer on the OpenCL device
    cl_actualIDsTemp = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_int));
    
    cout << "Created actual ID temp buffer \n";
    
    // Create finished buffer on the OpenCL device
    cl_finished = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_uchar));
    
    cout << "Created finished buffer \n";
    
    // Create material index buffer on the OpenCL device
    cl_mtlidxs = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_int));
    
    cout << "Created material index buffer \n";
    
    // Create ray buffer on the OpenCL device
    cl_rays = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * 3 * sizeof(cl_float3));
    
    cout << "Created ray buffer \n";
    
    // Create useful nums buffer on the OpenCL device
    cl_usefulnums = Buffer(context, CL_MEM_READ_ONLY, 11 * sizeof(cl_uint));
    queue.enqueueWriteBuffer(cl_usefulnums, CL_TRUE, 0, 11 * sizeof(cl_uint), cpu_usefulnums);
    
    cout << "Wrote useful numbers \n";
    
    // Create random buffer on the OpenCL device
    cl_randoms = Buffer(context, CL_MEM_READ_WRITE, window_width * window_height * sizeof(cl_uint));
    queue.enqueueWriteBuffer(cl_randoms, CL_TRUE, 0, window_width * window_height * sizeof(cl_uint), cpu_randoms);
    
    cout << "Wrote randoms \n";
    
    // Create ibl buffer on the OpenCL device
    cl_ibl = Buffer(context, CL_MEM_READ_ONLY, ibl_width * ibl_height * sizeof(cl_float3));
    queue.enqueueWriteBuffer(cl_ibl, CL_TRUE, 0, ibl_width * ibl_height * sizeof(cl_float3), cpu_ibl);
    
    cout << "Wrote IbL \n";
    
    // Create sphere buffer on the OpenCL device
    cl_spheres = Buffer(context, CL_MEM_READ_ONLY, sphere_amt * sizeof(Sphere));
    queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_amt * sizeof(Sphere), cpu_spheres);
    
    cout << "Wrote spheres \n";
    
    // Create triangle buffer on the OpenCL device
    cl_triangles = Buffer(context, CL_MEM_READ_ONLY, triangle_amt * sizeof(Triangle));
    queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, triangle_amt * sizeof(Triangle), cpu_triangles);
    
    cout << "Wrote triangles \n";
    
    // Create BVH node buffer on the OpenCL device
    cl_nodes = Buffer(context, CL_MEM_READ_ONLY, bvhnode_amt * sizeof(BVHNode));
    queue.enqueueWriteBuffer(cl_nodes, CL_TRUE, 0, bvhnode_amt * sizeof(BVHNode), cpu_bvhs);
    
    cout << "Wrote BVH \n";
    
    // Create material buffer on the OpenCL device
    cl_materials = Buffer(context, CL_MEM_READ_ONLY, material_amt * sizeof(Material));
    queue.enqueueWriteBuffer(cl_materials, CL_TRUE, 0, material_amt * sizeof(Material), cpu_materials);
    
    cout << "Wrote materials \n";
    
    // Create medium buffer on the OpenCL device
    cl_mediums = Buffer(context, CL_MEM_READ_ONLY, medium_amt * sizeof(Medium));
    queue.enqueueWriteBuffer(cl_mediums, CL_TRUE, 0, medium_amt * sizeof(Medium), cpu_mediums);
    
    cout << "Wrote mediums \n";
    
    // Create camera buffer on the OpenCL device
    cl_camera = Buffer(context, CL_MEM_READ_ONLY, sizeof(Camera));
    queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(Camera), cpu_camera);
    
    cout << "Wrote camera \n";
    
    // create OpenCL buffer from OpenGL vertex buffer object
    cl_vbo = BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
    cl_vbos.push_back(cl_vbo);
    
    cout << "Wrote VBO \n";
    
    // reserve memory buffer on OpenCL device to hold image buffer for accumulated samples
    cl_accumbuffer = Buffer(context, CL_MEM_WRITE_ONLY, window_width * window_height * sizeof(cl_float3));
    
    cout << "Created accumbuffer \n";
    
}

void initCLKernel(){
    
    // pick a rendermode
    unsigned int rendermode = 1;
    
    // Create a kernel (entry point in the OpenCL source program)
    kernel = Kernel(program, "render_kernel");
    
    // specify OpenCL kernel arguments
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
    kernel.setArg(10, voidcolor);
    kernel.setArg(11, cl_camera);
    kernel.setArg(12, framenumber);

}


void initInitKernel(){
    
    // Create a kernel (entry point in the OpenCL source program)
    init_kernel = Kernel(program, "init_kernel");
    
    // specify OpenCL kernel arguments
    init_kernel.setArg(0, cl_rays);
    init_kernel.setArg(1, cl_throughputs);
    init_kernel.setArg(2, cl_actualIDs);
    init_kernel.setArg(3, cl_randoms);
    init_kernel.setArg(4, cl_camera);
    init_kernel.setArg(5, window_width);
    init_kernel.setArg(6, window_height);
    init_kernel.setArg(7, bools);

}


void initIntersectionKernel() {
    
    // Create a kernel (entry point in the OpenCL source program)
    intersection_kernel = Kernel(program, "intersection_kernel");
    
    // specify OpenCL kernel arguments
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
}


void initShadingKernel() {
    
    // Create a kernel (entry point in the OpenCL source program)
    shading_kernel = Kernel(program, "shading_kernel");
    
    // specify OpenCL kernel arguments
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
}


void initRmoKernel() {
    
    // Create a kernel (entry point in the OpenCL source program)
    rmo_kernel = Kernel(program, "rmo_kernel");
    
    // specify OpenCL kernel arguments
    rmo_kernel.setArg(0, cl_actualIDs);
    rmo_kernel.setArg(1, cl_actualIDsTemp);
    rmo_kernel.setArg(2, cl_finished);
    rmo_kernel.setArg(3, window_width * window_height);

}


void initRmfKernel() {
    
    // Create a kernel (entry point in the OpenCL source program)
    rmf_kernel = Kernel(program, "rmf_kernel");
    
    // specify OpenCL kernel arguments
    rmf_kernel.setArg(0, cl_actualIDs);
    rmf_kernel.setArg(1, cl_actualIDsTemp);

}

void initFinalKernel() {
    
    // Create a kernel (entry point in the OpenCL source program)
    final_kernel = Kernel(program, "final_kernel");
    
    // specify OpenCL kernel arguments
    final_kernel.setArg(0, cl_vbo);
    final_kernel.setArg(1, cl_accumbuffer);
    final_kernel.setArg(2, window_width);
    final_kernel.setArg(3, window_height);
    final_kernel.setArg(4, framenumber);
}


void initCLKernels() {

    initInitKernel();

    cout << "Initialized Init Kernel\n";

    initIntersectionKernel();

    cout << "Initialized Intersection Kernel\n";

    initShadingKernel();

    cout << "Initialized Shading Kernel\n";

    initRmoKernel();

    cout << "Initialized Rm Kernel\n";

    initRmfKernel();

    cout << "Initialized Rmf Kernel\n";

    initFinalKernel();

    cout << "Initialized Final Kernel\n";

}



typedef struct Shift {
    int i;
    int f;
    int s;
    int a;

    Shift(int i_, int f_, int s_, int a_) : i(i_), f(f_), s(s_), a(a_) { }
} Shift;

void runKernels() {

    std::size_t cl_int_size = sizeof(cl_int);

    std::size_t global_work_size = window_width * window_height;
    std::size_t global_work_size_tmp = window_width * window_height;
    std::size_t local_work_size = init_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    
    // Ensure the global work size is a multiple of local work size
    if(global_work_size % local_work_size != 0)
        global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

    // cout << global_work_size << endl;
    // cout << local_work_size << endl;
    
    // Make sure OpenGL is done using the VBOs
    glFinish();
    
    // Pass in the vector of VBO buffer objects
    queue.enqueueAcquireGLObjects(&cl_vbos);
    queue.finish();

    // global_work_group_size = (cl_uint *)queue.enqueueMapBuffer(cl_globalworkgroupsize, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_uint));  
    // global_work_group_size[0] = window_width * window_height;
    // queue.enqueueUnmapMemObject(cl_globalworkgroupsize, global_work_group_size); 
    
    // Launch init kernel
    queue.enqueueNDRangeKernel(init_kernel, NULL, global_work_size, local_work_size);
    // queue.finish();
    int bounces = 1500;

    for(int n = 0; n < bounces && global_work_size > 0; n++) {

        // Launch intersection kernel
        queue.enqueueNDRangeKernel(intersection_kernel, NULL, global_work_size, local_work_size);
        // queue.finish();

        shading_kernel.setArg(16, n);

        // Launch shading kernel
        queue.enqueueNDRangeKernel(shading_kernel, NULL, global_work_size, local_work_size);
        queue.finish();

        // cout << *global_work_group_size << endl;
        // queue.finish();
        if(n < bounces - 1) {
            // Get working thread count
            // global_work_group_size = (cl_uint *)queue.enqueueMapBuffer(cl_globalworkgroupsize, CL_FALSE, CL_MAP_READ, 0, sizeof(cl_uint));
            // global_work_size = global_work_group_size[0];
            // if(global_work_size == 0) break;
            // cout << global_work_group_size[0] << endl;
            // queue.enqueueUnmapMemObject(cl_globalworkgroupsize, global_work_group_size);

            global_work_size = global_work_size_tmp;
            queue.enqueueReadBuffer(cl_finished, CL_FALSE, 0, global_work_size * sizeof(cl_uchar), cpu_finished);

            int global_work_size_old = global_work_size;

            int i = 0;
            int idx;
            bool allthewayout = false;

            // Start timer
            auto start = std::chrono::high_resolution_clock::now();
            
            int currentpos = 0;

            queue.finish();
            if(n == 0 || global_work_size == window_width * window_height) {

                while(i < global_work_size_old) {
                    int in_a_row = 0;
                    while(cpu_finished[i]) {
                        in_a_row++;
                        i++;
                        if(i == global_work_size_old) {
                            allthewayout = true;
                            break;
                        }
                    }
                    global_work_size -= in_a_row;

                    if(allthewayout) break;

                    // if(in_a_row) {
                    //     Shift ns(i, i - in_a_row - totalshift, in_a_row, global_work_size_old - i);
                    //     if(shifts.size() > 0) {
                    //         shifts[shifts.size() - 1].a -= ns.a + ns.s;
                    //     }
                    //     shifts.push_back(ns);
                    //     totalshift += in_a_row;
                    // }
                    cpu_actualIDs[currentpos] = i;
                    cpu_finished[currentpos] = cpu_finished[i];
                    i++;
                    currentpos++;
                }
            } else {
                // queue.enqueueReadBuffer(cl_actualIDs, CL_FALSE, 0, global_work_size * cl_int_size, cpu_actualIDs);

                // std::vector<Shift> shifts;
                // shifts.clear();
                // int totalshift = 0;

                while(i < global_work_size_old) {
                    int in_a_row = 0;
                    while(cpu_finished[i]) {
                        in_a_row++;
                        i++;
                        if(i == global_work_size_old) {
                            allthewayout = true;
                            break;
                        }
                    }
                    global_work_size -= in_a_row;

                    if(allthewayout) break;

                    // if(in_a_row) {
                    //     Shift ns(i, i - in_a_row - totalshift, in_a_row, global_work_size_old - i);
                    //     if(shifts.size() > 0) {
                    //         shifts[shifts.size() - 1].a -= ns.a + ns.s;
                    //     }
                    //     shifts.push_back(ns);
                    //     totalshift += in_a_row;
                    // }
                    cpu_actualIDs[currentpos] = cpu_actualIDs[i];
                    cpu_finished[currentpos] = cpu_finished[i];
                    i++;
                    currentpos++;
                }
                // auto finish = std::chrono::high_resolution_clock::now();
                
                // std::chrono::duration<double> elapsed = finish - start;

                // printf("Identified %d shifts in %f s.\n", shifts.size(), elapsed.count());


                // cout << "Shifting memory...\n";

                // for(Shift s : shifts) {
                //     // printf("Shifting %d indices from %d to %d.\n", s.a, s.i, s.f);
                //     // queue.enqueueWriteBuffer(cl_actualIDs, CL_FALSE, s.f * cl_int_size, s.a * cl_int_size, cpu_actualIDs + s.i);
                //     memmove(cpu_actualIDs + s.f, cpu_actualIDs + s.i, s.a * cl_int_size);
                // }

                // shifts.clear();

                // cout << "Shifted memory.\n";
            }


            if(global_work_size_old - global_work_size) {
                queue.enqueueWriteBuffer(cl_finished, CL_FALSE, 0, global_work_size * cl_int_size, cpu_finished);
                queue.enqueueWriteBuffer(cl_actualIDs, CL_FALSE, 0, global_work_size * cl_int_size, cpu_actualIDs);

                // Set global_work_size to kernel-computed value;
                global_work_size_tmp = global_work_size;
                local_work_size = init_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

                // Ensure the global work size is a multiple of local work size
                if(global_work_size % local_work_size != 0)
                    global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

                queue.finish();
            }
            // End timer
            auto finish = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> elapsed = finish - start;

            printf("Killed %d threads and reindexed and wrote the remainder in %f s.\n", global_work_size_old - global_work_size, elapsed.count());

        }

    }
    
    global_work_size = window_width * window_height;
    // local_work_size = init_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

    // Ensure the global work size is a multiple of local work size
    if (global_work_size % local_work_size != 0)
        global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

    // Launch final kernel
    queue.enqueueNDRangeKernel(final_kernel, NULL, global_work_size, local_work_size);
    queue.finish();
    // cout << global_work_group_size[0] << endl;
    
    //Release the VBOs so OpenGL can play with them
    queue.enqueueReleaseGLObjects(&cl_vbos);
    queue.finish();
    
}


void runKernel(){
    // every pixel in the image has its own thread or "work item",
    // so the total amount of work items equals the number of pixels
    std::size_t global_work_size = window_width * window_height;
    std::size_t local_work_size = 64;//kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    
    // Ensure the global work size is a multiple of local work size
    if (global_work_size % local_work_size != 0)
        global_work_size = (global_work_size / local_work_size + 1) * local_work_size;
    
    // Make sure OpenGL is done using the VBOs
    glFinish();
    
    // Pass in the vector of VBO buffer objects
    queue.enqueueAcquireGLObjects(&cl_vbos);
    queue.finish();
    
    // Launch the kernel
    queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size);
    queue.finish();
    
    //Release the VBOs so OpenGL can play with them
    queue.enqueueReleaseGLObjects(&cl_vbos);
    queue.finish();
}

void render() {

    if (buffer_reset){
        float arg = 0;
        queue.enqueueFillBuffer(cl_accumbuffer, arg, 0, window_width * window_height * sizeof(cl_float3));
        framenumber = 0;
    }
    buffer_reset = false;
    framenumber++;
    
    interactiveCamera->buildRenderCamera(cpu_camera);
    cpu_actualIDs = new cl_int[window_width * window_height];
    queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(Camera), cpu_camera);
    queue.enqueueFillBuffer(cl_actualIDs, CL_TRUE, 0, window_width * window_height * sizeof(cl_int));
    queue.enqueueFillBuffer(cl_finished, CL_TRUE, 0, window_width * window_height * sizeof(cl_uchar));
    queue.finish();
    
    // kernel.setArg(11, cl_camera);
    // kernel.setArg(12, framenumber - 1);
    init_kernel.setArg(4, cl_camera);
    final_kernel.setArg(4, framenumber - 1);
    
    // cout << "Running kernel...\n";

    // runKernel();
    runKernels();

    // cout << "Ran kernel.\n";
    
    drawGL();
    
    cout << samplesPerRun*framenumber << "\n";

}

void cleanUp(){
//    delete cpu_accumbuffer;
//    delete cpu_randoms;
//    delete cpu_usefulnums;
//    delete cpu_ibl;
//    delete cpu_spheres;
//    delete cpu_triangles;
//    delete cpu_bvhs;
//    delete cpu_materials;
//    delete cpu_mediums;
//    delete cpu_camera;
}

void initCamera() {
    delete interactiveCamera;
    
    Vec3 cam_pos = scn.cam_pos;
    Vec3 cam_fd = scn.cam_fd;
    Vec3 cam_up = scn.cam_up;
    float cam_focal_distance = scn.cam_focal_distance;
    float cam_aperture_radius = scn.cam_aperture_radius;
    
    interactiveCamera = new InteractiveCamera(cam_pos, cam_fd, cam_up, cam_focal_distance, cam_aperture_radius);
}

int main(int argc, char** argv){
    
    loadConfig("config", &window_width, &window_height, &scn_path, &interactive);
    
    cout << "Configurations loaded \n";
    
    initGL(argc, argv);
    
    cout << "OpenGL initialized \n";
    
    initOpenCL();
    
    cout << "OpenCL initialized \n";
    
    createVBO(&vbo);
    
    cout << "VBO created \n";
    
    Timer(0);
    
    cout << "Timer started \n";
    
    glFinish();
    
    cout << "glFinish executed \n";
    
    // Create Buffer Values
    createBufferValues();
    
    cout << "Buffer values created \n";
    
    // Write Buffer Values
    writeBufferValues();
    
    cout << "Buffer values written \n";
    
    // intitialize the kernel
    // initCLKernel();
    initCLKernels();
    
    cout << "CL Kernel initialized \n";
    
    // start rendering continuously
    glutMainLoop();
    
    cout << "glutMainLoop executed \n";
    
    // release memory
    cleanUp();
    
    return 0;
}
