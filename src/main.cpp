#define PI 3.141592653589793238f

#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <vector>
#include "bvh.h"
#include "camera.h"
#include "cl.hpp"
#include "cl_gl_interop.h"
#include "geometry.h"
#include "hdrloader/hdrloader.h"
#include "linear_algebra.h"
#include "material.h"
#include "objloader.h"
#include "user_interaction.h"

using namespace std;
using namespace cl;

const int samples = 2048;
const int samplesPerRun = 1;

// Image-based Lighting variables
int ibl_width;
int ibl_height;
cl_float4* cpu_ibl;

//cl_float4* cpu_accumbuffer;

// Sphere variables
Sphere* cpu_spheres;
int sphere_amt;

// Triangle variables
Triangle* cpu_triangles;
int triangle_amt;

// BVHNode variables
BVHNode* cpu_bvhs;
int bvhnode_amt;

// BVHNode variables
Material* cpu_materials;
int material_amt;

// Camera variable
Camera* cpu_camera = NULL;

// Host (CPU) buffer variables
cl_uint* cpu_randoms;
cl_uint* cpu_usefulnums;
// OpenCL boiler plate variables
Device device;
CommandQueue queue;
Kernel kernel;
Context context;
Program program;

// Device (CPU/GPU) Buffers
Buffer cl_output;
Buffer cl_randoms;
Buffer cl_ibl;
Buffer cl_spheres;
Buffer cl_triangles;
Buffer cl_nodes;
Buffer cl_materials;
Buffer cl_camera;
Buffer cl_usefulnums;
Buffer cl_accumbuffer;
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

void printErrorLog(const Program& program, const Device& device){
    
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
    queue = CommandQueue(context, device);
    
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
    
    // Build the program for the selected device
    cl_int result = program.build({ device }); // "-cl-fast-relaxed-math"
    if (result) cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
    if (result == CL_BUILD_PROGRAM_FAILURE) printErrorLog(program, device);
//    program = Program(context, kernel_source);
//    cl_int result = program.build({ device });
//    if (result == CL_BUILD_PROGRAM_FAILURE) {
//        // Check the build status
//        cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
//        if (status == CL_BUILD_ERROR) {
//            // Get the build log
//            string name = device.getInfo<CL_DEVICE_NAME>();
//            string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
//            cerr << "Build log for " << name << ":" << endl
//            << buildlog << endl;
//        }
//
//    }
    //          if (result) cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
    //    if (result == CL_BUILD_PROGRAM_FAILURE) {
    //        // Determine the size of the log
    //        char resultant[16384];
    //        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, resultant);
    //        printf("%s\n", resultant);
    
    // Create a kernel (entry point in the OpenCL source program)
//    kernel = Kernel(program, "render_kernel");
}

void buildScene() {
    
    vector<Material> materials;
    materials.clear();
    
    vector<Sphere> spheres;
    spheres.clear();

//    materials.push_back(Material(Vec3(0.9f, 0.3f, 0.3f)));//, Vec3(4.0f, 4.0f, 4.0f)));
//    Sphere sphere0(2.0f, Vec3(-9.0f, 2.00001f, 5.0f), 0);
//    spheres.push_back(sphere0);
//
//    materials.push_back(Material(Vec3(0.3f, 0.9f, 0.3f), 0.103f, 1.495f, 2));
//    Sphere sphere1(2.0f, Vec3(-3.0f, 2.00001f, 5.0f), 1);
//    spheres.push_back(sphere1);
//    mats[1].kd = Vec3(0.3f, 0.9f, 0.3f);
//    mats[1].ke = Vec3(0.0f, 0.0f, 0.0f);
//    mats[1].roughness = 0.356f;
//    mats[1].tex0 = -1;
//    mats[1].tex1 = -1;
//    mats[1].type = 0;
//
//    cpu_spheres[1].radius = 2.0f;
//    cpu_spheres[1].pos = Vec3(-3.0f, 2.00001f, 5.0f);
//    cpu_spheres[1].mtl = mats[1];
//
//    mats[2].kd = Vec3(0.3f, 0.3f, 0.9f);
//    mats[2].ke = Vec3(0.0f, 0.0f, 0.0f);
//    mats[2].tex0 = -1;
//    mats[2].tex1 = -1;
//    mats[2].type = 0;
//
//    cpu_spheres[2].radius = 2.0f;
//    cpu_spheres[2].pos = Vec3(3.0f, 2.00001f, 5.0f);
//    cpu_spheres[2].mtl = mats[2];
//    materials.push_back(Material(Vec3(0.612f, 0.851f, 0.694f), 0.05f, 1));
//    Sphere sphere2(2.0f, Vec3(3.0f, 2.00001f, 5.0f), 1);
//    spheres.push_back(sphere2);
//    cpu_spheres[1].radius = 2.0f;
//    cpu_spheres[1].pos = Vec3(3.0f, 2.00001f, 5.0f);
//    cpu_spheres[1].mtl = 1;
//
//    mats[2].kd = Vec3(0.3f, 0.3f, 0.9f);
//    mats[2].ke = Vec3(0.0f, 0.0f, 0.0f);
//    mats[2].roughness = 0.025f;//0.320936131f;
//    mats[2].ior = 1.3333333f;
//    mats[2].tex0 = -1;
//    mats[2].tex1 = -1;
//    mats[2].type = 2;
//
//    cpu_spheres[2].radius = 2.0f;
//    cpu_spheres[2].pos = Vec3(-3.0f, 2.00001f, 5.0f);
//    cpu_spheres[2].mtl = mats[2];
    
    vector<Triangle> triangles;
    triangles.clear();
    
    // (1 << 1) | 0000 = 0010
    // 0010 | 0000 = 0010
    // (0010 & 0010) >> 1
    
//    Material dragonMat;
//    dragonMat.kd = Vec3(0.7f, 0.3f, 0.3f);
//    dragonMat.ke = Vec3(0.0f, 0.0f, 0.0f);
//    dragonMat.roughness = 0.103f;//0.320936131f;
//    dragonMat.ior = 1.495f;
//    dragonMat.tex0 = -1;
//    dragonMat.tex1 = -1;
//    dragonMat.type = 2;
    
//    Material deerMat;
//    deerMat.kd = Vec3(0.3f, 0.3f, 0.9f);//Vec3(0.35f, 0.32f, 0.3f);
//    deerMat.ke = Vec3(0.0f, 0.0f, 0.0f);
//    deerMat.roughness = 0.00125f;//0.320936131f;
//    deerMat.ior = 1.3333333f;
//    deerMat.tex0 = -1;
//    deerMat.tex1 = -1;
//    deerMat.type = 2;

    Material emitMat(Vec3(1.0f, 1.0f, 1.0f), Vec3(4.0f, 4.0f, 4.0f));
    materials.push_back(emitMat);
    spheres.push_back(Sphere(1.0f, Vec3(-3.0f, 1.00001f, 9.0f), 0));
    spheres.push_back(Sphere(1.0f, Vec3(-3.0f, 1.00001f, 1.0f), 0));
    spheres.push_back(Sphere(1.0f, Vec3(-7.0f, 1.00001f, 5.0f), 0));
    spheres.push_back(Sphere(1.0f, Vec3(1.0f, 1.00001f, 5.0f), 0));
    
    vector<BVHNode> nodes;

    
//    loadObj(triangles, "cube", cubeMat, /*Vec3(-3.0f, 3.00001f, 5.0f)*/Vec3(-3.0f, 1.5f, 5.0f), Vec3(3.0f, 3.0f, 3.0f));
    int dragonMat = materials.size();
    materials.push_back(Material(Vec3(0.7f, 0.3f, 0.3f), 0.103f, 1.495f, 2));
    loadObj(triangles, "dragon", dragonMat, /*Vec3(-3.0f, 3.00001f, 5.0f)*/Vec3(-3.0f, 0.0f, 5.0f), Vec3(0.4f, 0.4f, 0.4f));
//    int deerMat = materials.size();
//    materials.push_back(Material(Vec3(0.3f, 0.3f, 0.9f), 0.103f, 1.3333333f, 1));
//    loadObj(triangles, "deer", deerMat, /*Vec3(-3.0f, 3.00001f, 5.0f)*/Vec3(-3.0f, 0.0f, 5.0f), Vec3(0.25f, 0.25f, 0.25f));

    nodes = build(triangles);
    
    sphere_amt = spheres.size();
    cout << sphere_amt << endl;
    
    triangle_amt = triangles.size();
    cout << triangle_amt << endl;
    
    bvhnode_amt = nodes.size();
    cout << bvhnode_amt << endl;
    
    material_amt = materials.size();
    cout << material_amt << endl;
    
    cpu_spheres = new Sphere[sphere_amt];
    for(int i = 0; i < sphere_amt; i++) {
        Sphere sr = spheres[i];
        cpu_spheres[i] = spheres[i];
    }
    spheres.clear();
    
    cpu_triangles = new Triangle[triangle_amt];
    for(int i = 0; i < triangle_amt; i++) {
        Triangle tr = triangles[i];
        cpu_triangles[i] = triangles[i];
    }
    triangles.clear();
    
    cpu_bvhs = new BVHNode[bvhnode_amt];
    for(int i = 0; i < bvhnode_amt; i++) {
        BVHNode nd = nodes[i];
        cpu_bvhs[i] = nodes[i];
    }
    nodes.clear();
    
    cpu_materials = new Material[material_amt];
    for(int i = 0; i < material_amt; i++) {
        Material ml = materials[i];
        cpu_materials[i] = materials[i];
    }
    materials.clear();
    
}

void createBufferValues() {
    
    // Assemble Useful Numbers
    cpu_usefulnums = new cl_uint[9];
    cpu_usefulnums[0] = (cl_uint) window_width;
    cpu_usefulnums[1] = (cl_uint) window_height;
    
    default_random_engine generator;
    uniform_int_distribution<int> distribution(1, (1 << 31) - 2);
    
    // Generate random seeds
    cpu_randoms = new cl_uint[window_width * window_height];
    for(int i = 0; i < window_width * window_height; i++) {
        cpu_randoms[i] = (cl_uint) (distribution(generator));
    }
    
//    cpu_accumbuffer = new cl_float3[window_width * window_height];
    
    // IBL Loading
    string ibl_src_str = "res/Frozen_Waterfall_Ref.hdr";
    const char* ibl_src = ibl_src_str.c_str();
    
    HDRLoaderResult result;
    bool ret = HDRLoader::load(ibl_src, result);
    ibl_width = result.width;
    ibl_height = result.height;
    cpu_ibl = new cl_float3[ibl_width * ibl_height];
    float r, g, b;
    for(int i = 0; i < ibl_width * ibl_height; i++) {
//        r = pow(result.cols[3*i], 1.0f/2.2f);
//        g = pow(result.cols[3*i+1], 1.0f/2.2f);
//        b = pow(result.cols[3*i+2], 1.0f/2.2f);
        r = result.cols[3*i];
        g = result.cols[3*i+1];
        b = result.cols[3*i+2];
        cpu_ibl[i] = (cl_float3) {{r, g, b}};
    }
    cpu_usefulnums[2] = (cl_uint) ibl_width;
    cpu_usefulnums[3] = (cl_uint) ibl_height;
    
    // Construct the scene
    buildScene();
    cpu_usefulnums[4] = (cl_uint) sphere_amt;
    cpu_usefulnums[5] = (cl_uint) triangle_amt;
    cpu_usefulnums[6] = (cl_uint) bvhnode_amt;
    cpu_usefulnums[7] = (cl_uint) material_amt;
    cpu_usefulnums[8] = (cl_uint) samplesPerRun;
    
    initCamera();
    
    cpu_camera = new Camera();
    interactiveCamera->buildRenderCamera(cpu_camera);
    
}

void writeBufferValues() {
    
    // Create useful nums buffer on the OpenCL device
    cl_usefulnums = Buffer(context, CL_MEM_READ_ONLY, 9 * sizeof(cl_uint));
    queue.enqueueWriteBuffer(cl_usefulnums, CL_TRUE, 0, 9 * sizeof(cl_uint), cpu_usefulnums);
    
    // Create random buffer on the OpenCL device
    cl_randoms = Buffer(context, CL_MEM_READ_WRITE, window_width * window_height * sizeof(cl_uint));
    queue.enqueueWriteBuffer(cl_randoms, CL_TRUE, 0, window_width * window_height * sizeof(cl_uint), cpu_randoms);
    
    cout << "Wrote randoms \n";
    
    // Create ibl buffer on the OpenCL device
    cl_ibl = Buffer(context, CL_MEM_READ_ONLY, ibl_width * ibl_height * sizeof(cl_float3));
    queue.enqueueWriteBuffer(cl_ibl, CL_TRUE, 0, ibl_width * ibl_height * sizeof(cl_float3), cpu_ibl);
    
    cout << "Wrote IbL \n";
    
    // Create scene buffer on the OpenCL device
    cl_spheres = Buffer(context, CL_MEM_READ_ONLY, sphere_amt * sizeof(Sphere));
    queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_amt * sizeof(Sphere), cpu_spheres);
    
    cout << "Wrote spheres \n";
    
    // Create scene buffer on the OpenCL device
    cl_triangles = Buffer(context, CL_MEM_READ_ONLY, triangle_amt * sizeof(Triangle));
    queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, triangle_amt * sizeof(Triangle), cpu_triangles);
    
    cout << "Wrote triangles \n";
    
    // Create scene buffer on the OpenCL device
    cl_nodes = Buffer(context, CL_MEM_READ_ONLY, bvhnode_amt * sizeof(BVHNode));
    queue.enqueueWriteBuffer(cl_nodes, CL_TRUE, 0, bvhnode_amt * sizeof(BVHNode), cpu_bvhs);
    
    cout << "Wrote BVH \n";
    
    // Create scene buffer on the OpenCL device
    cl_materials = Buffer(context, CL_MEM_READ_ONLY, material_amt * sizeof(Material));
    queue.enqueueWriteBuffer(cl_materials, CL_TRUE, 0, material_amt * sizeof(Material), cpu_materials);
    
    cout << "Wrote materials \n";
    
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
    //kernel.setArg(0, cl_output);
    kernel.setArg(0, cl_accumbuffer);
    kernel.setArg(1, cl_usefulnums);
    kernel.setArg(2, cl_randoms);
    kernel.setArg(3, cl_ibl);
    kernel.setArg(4, cl_vbo);
    kernel.setArg(5, cl_spheres);
    kernel.setArg(6, cl_triangles);
    kernel.setArg(7, cl_nodes);
    kernel.setArg(8, cl_materials);
    kernel.setArg(9, cl_camera);
    kernel.setArg(10, framenumber);

}

void runKernel(){
    // every pixel in the image has its own thread or "work item",
    // so the total amount of work items equals the number of pixels
    std::size_t global_work_size = window_width * window_height;
    std::size_t local_work_size = 64;//kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    
    // Ensure the global work size is a multiple of local work size
    if (global_work_size % local_work_size != 0)
        global_work_size = (global_work_size / local_work_size + 1) * local_work_size;
    
    //Make sure OpenGL is done using the VBOs
    glFinish();
    
    //this passes in the vector of VBO buffer objects
    queue.enqueueAcquireGLObjects(&cl_vbos);
    queue.finish();
    
    // launch the kernel
    queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size); // local_work_size
    queue.finish();
    
    //Release the VBOs so OpenGL can play with them
    queue.enqueueReleaseGLObjects(&cl_vbos);
    queue.finish();
}

//inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
//
// // convert RGB float in range [0,1] to int in range [0, 255]
//inline int toInt(float x){ return int(clamp(x) * 255 + .5); }
//
//float filmicF(float f) {
//    float x = f;
//    x *= 0.6f;
//    x = max(0.0f, x - 0.004f);
//    x = (x*(6.2f*x+.5f))/(x*(6.2f*x+1.7f)+0.06f);
//    return x;
//}
//
//void saveImage(){
//
//    queue.enqueueReadBuffer(cl_accumbuffer, CL_TRUE, 0, window_width * window_height * sizeof(cl_float3), cpu_accumbuffer);
//    queue.finish();
//
//    FILE *f = fopen("opencl_raytracer.ppm", "w");
//    fprintf(f, "P3\n%d %d\n%d\n", window_width, window_height, 255);
//
//    for (int i = 0; i < window_width * window_height; i++) {
//        int pixel_y = i / window_width;
//        int pixel_x = i % window_width;
//        int pos = (window_height - pixel_y)*window_width + pixel_x;
//        fprintf(f, "%d %d %d ",
//                toInt(filmicF(cpu_accumbuffer[pos].s[0])),
//                toInt(filmicF(cpu_accumbuffer[pos].s[1])),
//                toInt(filmicF(cpu_accumbuffer[pos].s[2])));
//    }
//
//    cout << "Image saved." << endl;
//}

void render() {
    
//    if(framenumber % 16) {
//        saveImage();
//    }
    
    //cpu_spheres[1].position.y += 0.01f;
//    queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_amt * sizeof(Sphere), cpu_spheres);
//    queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, triangle_amt * sizeof(Triangle), cpu_triangles);
//    queue.enqueueWriteBuffer(cl_nodes, CL_TRUE, 0, bvhnode_amt * sizeof(BVHNode), cpu_bvhs);
//    queue.enqueueWriteBuffer(cl_materials, CL_TRUE, 0, material_amt * sizeof(Material), cpu_materials);
    
    
//    for(int i = 0; i < window_width * window_height; i++) {
//        cpu_randoms[i] = (cl_uint) (rand()*4294967295);
//    }
//    queue.enqueueWriteBuffer(cl_randoms, CL_TRUE, 0, window_width * window_height * sizeof(cl_uint), cpu_randoms);
    
    if (buffer_reset){
        float arg = 0;
        queue.enqueueFillBuffer(cl_accumbuffer, arg, 0, window_width * window_height * sizeof(cl_float3));
//        queue.enqueueWriteBuffer(cl_randoms, arg, 0, window_width * window_height * sizeof(cl_uint), cpu_randoms);
        framenumber = 0;
    }
    buffer_reset = false;
    framenumber++;
    
    interactiveCamera->buildRenderCamera(cpu_camera);
    queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(Camera), cpu_camera);
    queue.finish();
    
    // kernel.setArg(0, cl_spheres);  //  works even when commented out/
    kernel.setArg(9, cl_camera);
    kernel.setArg(10, framenumber - 1);
    
    runKernel();
    
//    if(framenumber % 16 == 0 && framenumber != 0) {
//        saveImage();
//    }
    
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
//    delete cpu_camera;
}

void initCamera() {
    delete interactiveCamera;
    Vec3 cam_pos = Vec3(4.216578948221484f, 2.375f, 0.34339889486771863f);
    Vec3 cam_fd = Vec3(-0.7156478248575902f, -0.05930652721420354f, 0.6959388813727766f);
    Vec3 cam_up = Vec3(-0.042517570286490336f, 0.9982398187959389f, 0.04134634672114762f);
    float cam_focal_distance = 7.370589916300956f;
    float cam_aperture_radius = 5e-2f;
    interactiveCamera = new InteractiveCamera(cam_pos, cam_fd, cam_up, cam_focal_distance, cam_aperture_radius);
}

int main(int argc, char** argv){
    
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
    initCLKernel();
    
    cout << "CL Kernel initialized \n";
    
    /*for(int i = 0; i < 1024; i++) {
        render();
    }
    
    saveImage();*/
    
    // start rendering continuously
    glutMainLoop();
    
    cout << "glutMainLoop executed \n";
    
    // release memory
    cleanUp();
    
    return 0;
}
