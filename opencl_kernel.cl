__constant const float eps = 3e-5f;
__constant const float INF = 1e20f;
__constant const float PI = 3.14159265358979324f;
__constant const float invPI = 0.31830988618379067f;
__constant const float inv4PI = 0.07957747154594767f;
__constant const float PI2 = 6.28318530717958647f;
__constant const bool bilerp = false;
__constant const float cba = 0.30901699437494742f;
__constant const float sba = 0.951056516295153572f;
__constant const float3 ZENITH_DIR = (float3)(1000.0f, 500.0f, -500.0f);
__constant const float3 NORMALIZED_ZENITH_DIR = (float3)(0.81649658f, 0.40824829f, -0.40824829f);

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

typedef struct Ray {
    float3 origin;
    float3 dir;
    float3 inv_dir;
} Ray;

typedef struct BVHNode {
    float3 box[2];
    int parent;
    int child1;
    int child2;
    int isLeaf;
} BVHNode;

typedef struct Medium {
    float3 absCoefficient;
    float scatterCoefficient;
    int dummy[3];
} Medium;

typedef struct Material {
    float3 kd;
    float3 ke;
    float roughness;
    float ior;
    int tex0;
    int tex1;
    int type;
    int medIdx;
    int dummy[2];
} Material;

__constant Material ground = {(float3)(0.9f, 0.908f, 0.925f), (float3)(0.0f, 0.0f, 0.0f), 0.00125f, 1.0f, -1, -1, 1};

typedef struct Sphere {
    float3 pos;
    float radius;
    int mtlidx;
    int dummy[2];
} Sphere;

typedef struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    int mtlidx;
    int dummy[3];
} Triangle;

typedef struct Camera {
    float3 pos;
    float3 fd;
    float3 up;
    float focal_distance;
    float aperture_radius;
    int blades;
    int dummy;
} Camera;

static float rand(unsigned int *seed) {
    unsigned int x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    return as_float((x & 0x007FFFFF) | 0x3F800000) - 1.0;
}

float3 tonemapFilmic(const float3 f) {
    float3 x = max(0.0f, f - 0.004f);
    float3 xm = 6.2f*x;
    return native_divide(x*(xm + 0.5f), x*(xm + 1.7f) + 0.06f);
}

float2 samplePoly(unsigned int *seed) {
    int triIdx = (int) (rand(seed) * 5.0f);
    
    float r1 = native_sqrt(rand(seed));
    
    float r2 = rand(seed);
    float t = PI*(0.5f + triIdx*0.4f);
    float ct = native_cos(t);
    float st = native_sin(t);
    float2 a0 = (float2)(ct, st);
    float2 a1 = (float2)(ct*cba - st*sba, st*cba + ct*sba);
    return (r1 * (a0 - r2 * (a0 - a1)));
}


Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height, const bool use_DOF,
                 unsigned int *seed, __constant const Camera* cam) {
    
    float offx = rand(seed);
    float offy = rand(seed);
    
    float ih = native_recip((float) height);
    
    float har = native_divide((float) width, (float) (height << 1));
    float x0 = ih * (x_coord + offx - 0.5f) - har;
    float y0 = 0.5f - ih * (y_coord + offy - 0.5f);
    
    const float3 fd = cam->fd;
    const float3 up = cam->up;
    const float3 rt = cross(up, fd);
    const float3 raydir = x0*rt + y0*up + fd;
    
    if(use_DOF) {
        Ray ray;
        float2 pt = cam->aperture_radius*samplePoly(seed);
        float3 aptOffset = up*pt.y + rt*pt.x;
        float3 dn = normalize(raydir*cam->focal_distance - aptOffset);
        ray.origin = cam->pos + aptOffset;
        ray.dir = dn;
        return ray;
    } else {
        Ray ray;
        ray.origin = cam->pos;
        ray.dir = normalize(raydir);
        return ray;
    }
}

float3 orientTo(float3 a, float3 N) {
    float sign = copysign(1.0f, N.z);
    const float f = -native_recip(sign + N.z);
    const float g = N.x * N.y * f;
    const float3 u = (float3)(1.0f + sign * N.x * N.x * f, sign * g, -sign * N.x);
    const float3 v = (float3)(g, sign + N.y * N.y * f, -N.y);
    return a.x * u + a.y * v + a.z * N;
}

float3 cosineRandHemi(const float3 N, const float u1, const float u2) {
    float r1 = 2.0f * 3.141592653f * u1;
    const float r2 = u2;
    const float r2s = native_sqrt(r2);
    float x = r2s * native_cos(r1);
    float y = r2s * native_sin(r1);
    float z = native_sqrt(max(0.0f, 1.0f - r2));
    float sign = copysign(1.0f, N.z);
    const float a = -native_recip(sign + N.z);
    const float b = N.x * N.y * a;
    const float3 u = (float3)(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    const float3 v = (float3)(b, sign + N.y * N.y * a, -N.y);
    return x * u + y * v + z * N;
}

bool intersect_sphere(__global Sphere* sphere, const Ray* ray, float3* point, float3* normal, float* t) {
    float3 rayToCenter = sphere->pos - ray->origin;
    float b = dot(rayToCenter, ray->dir);
    float c = dot(rayToCenter, rayToCenter) - sphere->radius*sphere->radius;
    float d = b * b - c;
    
    if (d <= eps) {
        return false;
    }
    d = native_sqrt(d);
    float temp = b - d;
    if(temp > *t) return false;
    
    if (temp < eps) {
        temp = b + d;
        if (temp < eps || temp > *t) return false;
    }
    
    *t = temp;
    return true;
}

bool intersect_triangle(__read_only image1d_t triangles_img, const int idx, const Ray* ray,
                        float3* point, float3* normal, float* t, const bool cull) {
    
    float3 v0 = read_imagef(triangles_img, sampler, idx).xyz;
    float3 v1 = read_imagef(triangles_img, sampler, idx + 1).xyz;
    float3 v2 = read_imagef(triangles_img, sampler, idx + 2).xyz;
    float3 pvec = cross(ray->dir, v2);
    
    /* Backface culling */
    /*if(det < eps && false) return false;*/
    
    float invDet = native_recip(dot(v1, pvec));
    
    float3 tvec = ray->origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if(u < 0 || u > 1) return false;
    
    float3 qvec = cross(tvec, v1);
    float v = dot(ray->dir, qvec) * invDet;
    if(v < 0 || u + v > 1) return false;
    
    float temp = dot(v2, qvec) * invDet;
    if(temp < eps || temp > *t) return false;
    
    *point = v1;
    *normal = v2;
    *t = temp;
    return true;
}

__constant const float3 GROUND_NORM = (float3)(0.0f, 1.0f, 0.0f);

bool intersect_ground(const Ray* ray, float3* point, float3* normal, float* t) {
    float temp = -ray->origin.y * ray->inv_dir.y;
    if(temp < eps || temp > *t) return false;
    float3 p = temp * ray->dir + ray->origin;
    if(p.x > 40.0 || p.x < -40.0 || p.z > 40.0 || p.z < -40.0)
        return false;
    *t = temp;
    *point = p;
    *normal = ray->dir.y < 0.0f ? GROUND_NORM : -GROUND_NORM;
    return true;
}

bool intersect_aabb(const float3 min, const float3 max, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (min - r->origin) * invD;
    float3 t1s = (max - r->origin) * invD;
    
    float3 tsmaller = fmin(t0s, t1s);
    float3 tbigger = fmax(t0s, t1s);
    
    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    float tmax = fmin(*t, fmin(tbigger[0], fmin(tbigger[1], tbigger[2])));
    
    return (tmin <= tmax);
}

bool intersect_aabb_lw(__global BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box[0] - r->origin) * invD;
    float3 t1s = (b->box[1] - r->origin) * invD;
    
    float3 tsmaller = fmin(t0s, t1s);
    
    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    
    return (tmin <= *t);
}

float intersect_aabb_dist(const float3 min, const float3 max, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (min - r->origin) * invD;
    float3 t1s = (max - r->origin) * invD;
    
    float3 tsmaller = fmin(t0s, t1s);
    float3 tbigger = fmax(t0s, t1s);
    
    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    float tmax = fmin(*t, fmin(tbigger[0], fmin(tbigger[1], tbigger[2])));
    
    return (tmin <= tmax) ? tmin : INF;
}

void intersect_bvh(const Ray* ray, float3* point, float3* normal, float* t, int* triangle_id, int* sphere_id,
                   __read_only image1d_t node_min_img, __read_only image1d_t node_max_img,
                   __read_only image1d_t node_info_img, __read_only image1d_t triangles_img) {
    
    if(!intersect_aabb(read_imagef(node_min_img, sampler, 0).xyz,
                       read_imagef(node_max_img, sampler, 0).xyz,
                       ray, t))
        return;
    
    int4 current = read_imagei(node_info_img, sampler, 0);
    
    int currentIdx = 0;
    int lastIdx = -1;
    unsigned int depth = 0;
    unsigned int branch = 0;
    bool goingUp = false;
    bool swapped;
    int child1;
    int child2;
    int child1_mod;
    int child2_mod;
    
    while(true) {
        if(goingUp) {
            if(currentIdx == 0)
                return;
            
            lastIdx = currentIdx;
            currentIdx = current.x;
            branch &= (2 << depth) - 1;
            depth--;
        } else {
            depth++;
        }
        
        current = read_imagei(node_info_img, sampler, currentIdx);
        
        /*
         BVH Traversal: Either of the first two conditions are true if currently moving up the tree
         */
        child1 = current.y;
        child2 = current.z;
        
        goingUp = false;
        
        if(currentIdx < lastIdx) {
            swapped = (2 << depth) & branch;
            child1_mod = swapped ? child2 : child1;
            child2_mod = swapped ? child1 : child2;
            
            if(lastIdx == child2_mod) {
                /*
                 If done parsing right node, done with entire branch
                 */
                goingUp = true;
                
            } else {
                lastIdx = currentIdx;
                currentIdx = child2_mod;
            }
            continue;
        }
        
        if(current.w != 1) {
            float dist1 = intersect_aabb_dist(read_imagef(node_min_img, sampler, child1).xyz,
                                              read_imagef(node_max_img, sampler, child1).xyz,
                                              ray, t);
            float dist2 = intersect_aabb_dist(read_imagef(node_min_img, sampler, child2).xyz,
                                              read_imagef(node_max_img, sampler, child2).xyz,
                                              ray, t);
            bool hit1 = dist1 != INF;
            bool hit2 = dist2 != INF;
            
            if(hit1 || hit2) {
                lastIdx = currentIdx;
                
                bool reverse = dist2 < dist1;
                bool swapping = hit1 && (reverse || !hit2);
                branch |= (swapping << 1) << depth;
                currentIdx = (hit2 && reverse) ? child2 : child1;
                continue;
            }
            
            goingUp = true;
        } else {
            
            for (int i = child1; i < child2; i++)  {
                
                if(intersect_triangle(triangles_img, 3 * i, ray, point, normal, t, false)) {
                    *triangle_id = i;
                    *sphere_id = -1;
                }
                
            }
            
            goingUp = true;
        }
    }
}

bool intersect_scene(__global Sphere* spheres, __constant Material* materials, const Ray* ray,
                     float3* point, float3* normal, float* t, Material* m, const unsigned int sphere_count,
                     const unsigned int node_count, const unsigned int material_count, const bool use_ground,
                     __read_only image1d_t node_min_img, __read_only image1d_t node_max_img,
                     __read_only image1d_t node_info_img, __read_only image1d_t triangles_img,
                     __read_only image1d_t trimtlidx_img) {
    
    float ti = *t;

    if(use_ground)
        intersect_ground(ray, point, normal, t);
    
    int sphere_id = -1;
    
    for (unsigned int i = 0; i < sphere_count; i++)  {
        
        if(intersect_sphere(&spheres[i], ray, point, normal, t)) {
            sphere_id = i;
        }
    }
    
    int triangle_id = -1;
    
    if(node_count > 0)
        intersect_bvh(ray, point, normal, t, &triangle_id, &sphere_id, node_min_img,
                      node_max_img, node_info_img, triangles_img);
    
    if(sphere_id != -1) {
        int i = sphere_id;
        *point = ray->origin + (*t)*ray->dir;
        *normal = native_divide(*point - spheres[i].pos, spheres[i].radius);
        *m = materials[spheres[i].mtlidx];
    } else if(triangle_id != -1) {
        int i = triangle_id;
        *normal = normalize(cross(*point, *normal));
        *point = ray->origin + (*t)*ray->dir;
        *m = materials[read_imagei(trimtlidx_img, sampler, triangle_id).x];
    }
    
    return *t < ti;
}

float3 bilerp_func(float3 a, float3 b, float3 c, float3 d, float x, float y) {
    float g = 1.0f - x, h = 1.0f - y;
    float i = g*h, j = x*h, k = g*y, l = x*y;
    return a*i + b*j + c*k + d*l;
}

float3 sampleImage(float u, float v, int width, int height, __global float3* img) {
    if(bilerp) {
        float px = width * u - 0.5f;
        float py = height * v - 0.5f;
        int p0x = (int) px;
        int p0y = (int) py;
        int p1x = (p0x + 1) % width;
        int p2y = (p0y + 1) % height;
        float3 s0 = img[p0y*width + p0x];
        float3 s1 = img[p0y*width + p1x];
        float3 s2 = img[p2y*width + p0x];
        float3 s3 = img[p2y*width + p1x];
        return bilerp_func(s0, s1, s2, s3, px - p0x, py - p0y);
    }
    int x = (int) (u * width);
    int y = (int) (v * height);
    return img[y*width + x];
}

void diffuse_brdf(unsigned int *seed, float3 n, float3* wr) {
    float u1 = rand(seed);
    float u2 = rand(seed);
    *wr = normalize(cosineRandHemi(n, u1, u2));
}


/*
 
 GGX Functions
 
 */
float GGX_G1(float3 v, float3 m, float3 n, float a_g) {
    float cosTheta = dot(v, n);
    if(dot(v, m)*cosTheta <= 0.0f) {
        return 0.0f;
    }
    float tan2Theta = native_recip(cosTheta*cosTheta) - 1.0f;
    return native_divide(2.0f, 1.0f + native_sqrt(1.0f + a_g*a_g*tan2Theta));
}

float GGX_G(float3 i, float3 o, float3 m, float3 n, float a_g) {
    return (GGX_G1(i, m, n, a_g) * GGX_G1(o, m, n, a_g));
}


/*
 
 Plastic BRDF Functions
 
 */

float plastic_F(float3 i, float3 m) {
    float c = fabs(dot(i, m));
    float g = 1.1025f + c * c;
    if(g < 0.0f) {
        return 1.0f;
    }
    g = native_sqrt(g);
    float k = native_divide(c * (g + c) - 1.0f, c * (g - c) + 1.0f);
    float a = native_divide(0.5f * (g - c) * (g - c), (g + c) * (g + c));
    float b = 1.0f + k * k;
    return (a * b);
}

void plastic_brdf(unsigned int *seed, float3 n, float3 wo, float3* kd, float3* wr,
                  float* brdf, float roughness) {
    float3 i = wo;
    float a_g = roughness;
    
    float epsilon = rand(seed);
    if(epsilon == 1.0f) {
        epsilon = 0.9999f;
    }
    
    float xt = a_g*native_sqrt(native_divide(epsilon, 1.0f - epsilon));
    float ct = native_rsqrt(1.0f + xt*xt);
    float st = xt*ct;
    float phi = 2.0f*PI*rand(seed);
    
    float cp = native_cos(phi);
    float sp = native_sin(phi);
    float3 mlocal = (float3)(st*cp, st*sp, ct);
    float3 m = orientTo(mlocal, n);
    
    float fres = plastic_F(i, m);
    
    float idm = dot(i, m);
    
    if(rand(seed) <= fres) {
        *kd = (float3)(1.0f, 1.0f, 1.0f);
        *wr = normalize(2.0f*idm*m - i);
    } else {
        float u1 = rand(seed);
        float u2 = rand(seed);
        *wr = normalize(cosineRandHemi(n, u1, u2));
    }
    
    
    *brdf = fabs(native_divide(idm, dot(i, n)*ct)) * GGX_G(i, *wr, m, n, a_g);
    
}



/*
 
 Dielectric BRDF Functions
 
 */

float dielectric_F(float3 i, float3 m, float n_i, float n_t) {
    float c = fabs(dot(i, m));
    float g = native_divide(n_t*n_t, n_i*n_i) - 1.0f + c*c;
    if(g < 0.0f)
        return 1.0f;
    g = native_sqrt(g);
    float a0 = (g-c);
    float a1 = (g+c);
    float a = native_divide(a0*a0, 2.0f*a1*a1);
    float b0 = c*a1 - 1.0f;
    float b1 = c*a0 + 1.0f;
    float b = 1.0f + native_divide(b0*b0, b1*b1);
    return a*b;
}

void dielectric_brdf(unsigned int *seed, float3 n, float3 wo, float3* kd, float3 *ke ,float3* wr,
                     float* brdf, float ior, float roughness, bool* transmitted) {
    float3 i = wo;
    bool outside = dot(i, n) < 0;
    float n_o = 1.0f;
    float n_i = 1.0f;
    if(outside) {
        n_i = ior;
    } else {
        n_o = ior;
    }
    float a_g = roughness;
    float epsilon = rand(seed);
    float xt = a_g*native_sqrt(native_divide(epsilon, 1.0f - epsilon));
    float ct = native_rsqrt(1.0f + xt*xt);
    float st = xt*ct;
    float phi = 2.0f*PI*rand(seed);
    float cp = native_cos(phi);
    float sp = native_sin(phi);
    float3 mlocal = (float3)(st*cp, st*sp, ct);
    float3 m = orientTo(mlocal, n);
    
    float fres = dielectric_F(i, m, n_i, n_o);
    
    if(rand(seed) <= fres) {
        
        *kd = (float3)(1.0f, 1.0f, 1.0f);
        *ke = 0.0f * (*ke);
        
        float idm = dot(i, m);
        *wr = normalize(2.0f*idm*m - i);
        
        *brdf = fabs(native_divide(idm, dot(i, n)*ct)) * GGX_G(i, *wr, m, n, a_g);
    }
    else {
        
        *transmitted = true;
        
        float nr = native_divide(n_i, n_o);
        float idn = dot(i, n);
        float cr = dot(i, m);
        float k = 1.0f + nr*nr*(cr*cr - 1.0f);
        *wr = normalize(m*(nr*cr - copysign(1.0f, idn)*native_sqrt(k)) - i*nr);
        
        *brdf = fabs(native_divide(cr, idn*ct))*GGX_G(i, *wr, m, n, a_g);
    }
    
}


/*
 
 Mirror BRDF
 
 */
void mirror_brdf(float3 n, float3 wo, float3* wr) {
    *wr = 2.0f*dot(n, wo)*n - wo;
}


/*
 
 Generic BSDF function
 
 */
void bsdf(unsigned int *seed, float3 n, float3 wo, float3* wr, float3* kd, float3* ke,
          Material m, float* brdf, bool* transmitted) {
    
    int type = m.type;
    if(type == 0) {
        diffuse_brdf(seed, n, wr);
    }
    else if(type == 1) {
        plastic_brdf(seed, n, wo, kd, wr, brdf, m.roughness);
    }
    else if(type == 2) {
        dielectric_brdf(seed, n, wo, kd, ke, wr, brdf, m.ior, m.roughness, transmitted);
    }
    else if(type == 3) {
        mirror_brdf(n, wo, wr);
    }
    
}


float medSampleADist(Medium med, float maxDist, unsigned int *seed) {
    return -native_log(rand(seed)) / med.scatterCoefficient;
}

float medSampleDist(Medium med, float tFar, float weight, float volPDF, float prev, float* pdf) {
    float distance = prev;
    if(distance >= tFar) {
        *pdf = 1.0f;
        return tFar;
    }
    
    *pdf = native_exp(-med.scatterCoefficient * distance);
    return distance;
}

float3 medSampleScatterDir(float *volPDF, unsigned int *seed) {
    *volPDF = inv4PI;
    float theta = PI2 * rand(seed);
    float r1 = 1.0f - 2.0f * rand(seed);
    float sp = native_sqrt(1.0f - r1*r1);
    float x = sp * native_cos(theta);
    float y = sp * native_sin(theta);
    return (float3)(x, y, r1);
}

float medScatterDirPdf() {
    return inv4PI;
}

float3 medTransmission(Medium med, float distance) {
    float3 ad = -med.absCoefficient * distance;
    return native_exp(ad);
}


__constant float airScatterDist = 4000.0f;
__constant Medium air = {(float3)(0.0f, 0.0f, 0.0f), 1.0f / 4000.0f};


float3 trace(__global Sphere* spheres, __constant Material* materials, __constant Medium* mediums, const Ray* camray,
             const unsigned int sphere_count, const unsigned int node_count, const unsigned int material_count,
             const unsigned int medium_count, unsigned int *seed, int ibl_width, int ibl_height, __global float3* ibl,
             const float3 void_color, const bool use_IbL, const bool use_ground, __read_only image1d_t node_min_img,
             __read_only image1d_t node_max_img, __read_only image1d_t node_info_img, __read_only image1d_t triangles_img,
             __read_only image1d_t trimtlidx_img) {
    
    Ray ray = *camray;
    
    float3 throughput = (float3)(1.0f, 1.0f, 1.0f);
    float3 color = (float3)(0.0f, 0.0f, 0.0f);
    
    float3 point;
    float3 normal;
    float t;
    float brdf;
    float pdf;
    Material mtl;
    float3 wr;
    bool transmitted;
    int currMedIdx = -2;
    Medium med = air;
    bool hitSurface = false;
    float volWeight = 1.0f;
    float volPDF = 1.0f;
    
    for(int n = 0; n < 1500; n++) {
        
        mtl = ground;
        brdf = 1.0f;
        pdf = 1.0f;
        
        ray.inv_dir = native_recip(ray.dir);
        
        float maxDist = 1e16f;
        
        if(currMedIdx != -2) {
            maxDist = medSampleADist(med, maxDist, seed);
            t = maxDist;
        } else {
            t = maxDist;
        }
        
        bool hitThisTime = intersect_scene(spheres, materials, &ray, &point, &normal, &t, &mtl, sphere_count,
                                           node_count, material_count, use_ground, node_min_img, node_max_img,
                                           node_info_img, triangles_img, trimtlidx_img);
        
        hitSurface = true;
        
        if(!hitThisTime) {
            float3 env_map_pos = (float3)(0.0f, 15.0f, 0.0f);
            float3 eye = ray.origin - env_map_pos;
            float b = dot(eye, ray.dir);
            const float c = dot(eye, eye) - 1e4f;
            float d = b*b - c;
            t = native_sqrt(d) - b;
            hitSurface = false;
        }
        
        if(n > 8) {
            float p = min(0.95f, max(throughput.x, max(throughput.y, throughput.z)));
            if(rand(seed) > p)
                break;
            
            throughput *= native_recip(p);
        }
        
        if(currMedIdx != -2) {
            volWeight = 1.0f;
            float distance = medSampleDist(med, t, volWeight, volPDF, maxDist, &pdf);
            float3 transmission = medTransmission(med, distance);
            throughput *= transmission * volWeight;
            
            if(distance < t) {
                hitSurface = false;
                ray.origin += ray.dir*distance;
                
                float dirPdf = 0.0f;
                ray.dir = medSampleScatterDir(&pdf, seed);
                continue;
            }
        }
        
        if(hitSurface) {
            
            float3 kd = mtl.kd;
            float3 ke = mtl.ke;
            
            transmitted = false;
            
            bsdf(seed, normal, normalize(-1.0f * ray.dir), &wr, &kd, &ke, mtl, &brdf, &transmitted);
            
            color += throughput * ke;
            
            if(transmitted) {
                int mtlMed = mtl.medIdx;
                if(mtlMed != currMedIdx) {
                    currMedIdx = mtlMed;
                    med = mediums[mtlMed];
                } else {
                    currMedIdx = -2;
                    med = air;
                }
            }
            ray.origin = point;
            ray.dir = wr;
            throughput *= kd * brdf;
        } else {
            if(use_IbL) {
                /* image-based lighting */
                float3 env_map_pos = (float3)(0.0f, 15.0f, 0.0f);
                float3 eye = ray.origin - env_map_pos;
                float b = dot(eye, ray.dir);
                const float c = dot(eye, eye) - 1e16f;
                float d = b*b - c;
                
                const float3 smp = eye + ray.dir * (native_sqrt(d) - b);
                const float v = acospi(smp.y*1e-8f);
                const float u0 = 0.5f*atan2pi(smp.x, smp.z) + 1.0f;
                const float u = (u > 1.0f) ? u0 - 1.0f : u0;
                const float3 ibl_sample = sampleImage(u, v, ibl_width, ibl_height, ibl);
                return (color + throughput * void_color * ibl_sample);
            }
            else {
                return (color + throughput * void_color);
            }
            
            /*float cos2theta = dot(ray.dir, NORMALIZED_ZENITH_DIR);
             float costheta = native_sqrt(0.5f * (cos2theta + 1.0f));
             costheta = pow(costheta, 500.0f);
             float mult = max(130.0f * costheta + 1.0f, 1.0f);
             float3 multvec = (float3)(mult, mult, mult);
             float3 onevec = (float3)(1.0f, 1.0f, 1.0f);
             float3 gradientres = multvec * costheta + onevec * (1.0f - costheta);
             return color += throughput * gradientres;*/
            /*return color + throughput;*/
            /*return color;*/
        }
    }
    return color;
}

int pix_coord(const unsigned int width, const unsigned int height, int i) {
    int bigcol = i / (8 * height);
    int col = 8 * bigcol + i % 8;
    int rowcomp = i - (8 * bigcol * height);
    int row = rowcomp / 8;
    return (row * width + col);
}

union Color{ float c; uchar4 components; };

__kernel void render_kernel(__global float3* accumbuffer, __constant unsigned int* usefulnums,
                            __global unsigned int* randoms, __global float3* ibl, __global float3* output,
                            __global Sphere* spheres, __constant Material* materials, __constant Medium* mediums,
                            const float3 void_color, __constant const Camera* cam, int framenumber,
                            __read_only image1d_t node_min_img, __read_only image1d_t node_max_img,
                            __read_only image1d_t node_info_img, __read_only image1d_t triangles_img,
                            __read_only image1d_t trimtlidx_img) {
    
    const unsigned int width = usefulnums[0];
    const unsigned int height = usefulnums[1];
    const unsigned int ibl_width = usefulnums[2];
    const unsigned int ibl_height = usefulnums[3];
    const unsigned int sphere_amt = usefulnums[4];
    const unsigned int triangle_amt = usefulnums[5];
    const unsigned int node_amt = usefulnums[6];
    const unsigned int material_amt = usefulnums[7];
    const unsigned int medium_amt = usefulnums[8];
    const unsigned int samples = usefulnums[9];
    const unsigned int bools = usefulnums[10];
    const bool use_DOF = bools & 1;
    const bool use_IbL = bools & 2;
    const bool use_ground = bools & 4;
    const int wii = get_global_id(0);
    const int work_item_id = pix_coord(width, height, wii);
    unsigned int seed = randoms[work_item_id];
    unsigned int x_coord = work_item_id % width;    /* x-coordinate of the pixel */
    unsigned int y_coord = work_item_id / width;    /* y-coordinate of the pixel */
    
    float3 result = (float3)(0.0f, 0.0f, 0.0f);
    int spp = framenumber + 1;
    
    Ray camray = createCamRay(x_coord, height - y_coord, width, height, use_DOF, &seed, cam);
    
    float3 currres = trace(spheres, materials, mediums, &camray, sphere_amt, node_amt,
                           material_amt, medium_amt, &seed, ibl_width, ibl_height, ibl,
                           void_color, use_IbL, use_ground, node_min_img, node_max_img,
                           node_info_img, triangles_img, trimtlidx_img);
    
    if(isnan(currres).x == 0)
        result = currres;
    
    accumbuffer[work_item_id] += result;
    randoms[work_item_id] = seed;
    float3 res = tonemapFilmic(native_divide(accumbuffer[work_item_id], spp));
    
    union Color fcolor;
    fcolor.components = (uchar4)(convert_uchar3(clamp(res, 0.0f, 1.0f) * 255), 1);
    
    output[work_item_id] = (float3)(x_coord, y_coord, fcolor.c);
}
