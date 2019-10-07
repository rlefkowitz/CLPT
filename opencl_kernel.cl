__constant const float eps = 3e-5f;
__constant const float INF = 1e20f;
__constant const float PI = 3.14159265358979324f;
__constant const float invPI = 0.31830988618379067f;
__constant const bool DOF = true;
__constant const bool bilerp = false;
__constant const bool USE_BVH = true;
__constant const float cba = 0.30901699437494742f;
__constant const float sba = 0.951056516295153572f;
__constant const float3 ZENITH_DIR = (float3)(1000.0f, 500.0f, -500.0f);
__constant const float3 NORMALIZED_ZENITH_DIR = (float3)(0.81649658f, 0.40824829f, -0.40824829f);

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

typedef struct Material {
    float3 kd;
    float3 ke;
    float roughness;
    float ior;
    int tex0;
    int tex1;
    int type;
    int dummy[3];
} Material;

__constant Material ground = {(float3)(0.9f, 0.908f, 0.925f), (float3)(0.0f, 0.0f, 0.0f), 0.103f, 1.0f, -1, -1, 1};

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
    float3 x = max(0.0f, 0.6f*f - 0.004f);
    float3 xm = 6.2f*x;
    return (x*(xm + .5f))/(x*(xm + 1.7f) + 0.06f);
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
    return (r1 * (a0*(1.0f - r2) + a1*r2));
    return (r1 * (a0 - r2 * (a0 + a1));
}


Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height,
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
    
    if(DOF) {
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
    
    if (d <= eps) return false;
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

bool intersect_triangle(__global Triangle *triangle, const Ray* ray, float3* point, float3* normal, float* t, const bool cull) {
    
    /*float td = dot(ray->dir, triangle->vn);
     if(-td < eps && m->type != 2)
     return false;
     float temp = dot(triangle->v0 - ray->origin, triangle->vn) / td;
     if(temp >= *t || temp < eps) return false;
     
     float3 x = ray->origin + temp*ray->dir;
     
     float u = dot(triangle->vu, x);
     if(u < 0 || u > 1) return false;
     
     float v = dot(triangle->vv, x);
     if(v < 0 || u + v > 1) return false;
     
     *point = x;
     *t = temp;
     return true;*/
    
    float3 v0v1 = triangle->v1 - triangle->v0;
    float3 v0v2 = triangle->v2 - triangle->v0;
    float3 pvec = cross(ray->dir, v0v2);
    float det = dot(v0v1, pvec);
    
    /* Backface culling */
    /*if(det < eps && false) return false;*/
    
    float invDet = native_recip(det);
    
    float3 tvec = ray->origin - triangle->v0;
    float u = dot(tvec, pvec) * invDet;
    if(u < 0 || u > 1) return false;
    
    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray->dir, qvec) * invDet;
    if(v < 0 || u + v > 1) return false;
    
    float temp = dot(v0v2, qvec) * invDet;
    if(temp < eps || temp > *t) return false;
    
    *point = v0v1;
    *normal = v0v2;
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
    *normal = GROUND_NORM;
    return true;
}

bool intersect_aabb(const BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box[0] - r->origin) * invD;
    float3 t1s = (b->box[1] - r->origin) * invD;
    
    float3 tsmaller = fmin(t0s, t1s);
    float3 tbigger = fmax(t0s, t1s);
    
    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    float tmax = fmin(*t, fmin(tbigger[0], fmin(tbigger[1], tbigger[2])));
    
    return (tmin <= tmax);
}

bool intersect_aabb_lw(const BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box[0] - r->origin) * invD;
    float3 t1s = (b->box[1] - r->origin) * invD;
    
    float3 tsmaller = fmin(t0s, t1s);
    
    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    
    return (tmin <= *t);
}

float intersect_aabb_dist(const BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box[0] - r->origin) * invD;
    float3 t1s = (b->box[1] - r->origin) * invD;
    
    float3 tsmaller = fmin(t0s, t1s);
    float3 tbigger = fmax(t0s, t1s);
    
    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    float tmax = fmin(*t, fmin(tbigger[0], fmin(tbigger[1], tbigger[2])));
    
    return (tmin <= tmax) ? tmin : INF;
}

void intersect_bvh(__global Triangle* triangles, __global BVHNode* nodes, const Ray* ray, float3* point,
                   float3* normal, float* t, const unsigned int triangle_count, const unsigned int node_count,
                   int* triangle_id, int* sphere_id) {
    
    BVHNode current = nodes[0];
    
    if(!intersect_aabb(&current, ray, t))
        return;
    
    int currentIdx = 0;
    int lastIdx = -1;
    unsigned int depth = 0;
    unsigned int branch = 0;
    bool retrieved_current = true;
    bool goingUp = false;
    bool swapped;
    int child1;
    int child2;
    int child1_mod;
    int child2_mod;
    
    while(true) {
        if(goingUp) {
            if(currentIdx < 1)
                return;
            
            lastIdx = currentIdx;
            currentIdx = current.parent;
            branch &= (2 << depth) - 1;
            depth--;
        }
        
        if(!retrieved_current)
            current = nodes[currentIdx];
        retrieved_current = false;
        /*
         BVH Traversal: Either of the first two conditions are true if currently moving up the tree
         */
        child1 = current.child1;
        child2 = current.child2;
        
        goingUp = false;
        
        if(currentIdx > lastIdx) {
            if(current.isLeaf != 1) {
                BVHNode c1 = nodes[child1];
                BVHNode c2 = nodes[child2];
                float dist1 = intersect_aabb_dist(&c1, ray, t);
                float dist2 = intersect_aabb_dist(&c2, ray, t);
                bool hit1 = dist1 != INF;
                bool hit2 = dist2 != INF;
                
                if(hit1 && hit2) {
                    /*
                     Both distances are finite
                     */
                    retrieved_current = true;
                    
                    bool reverse = dist2 < dist1;
                    branch |= (reverse << 1) << depth;
                    
                    lastIdx = currentIdx;
                    currentIdx = reverse ? child2 : child1;
                    current = reverse ? c2 : c1;
                    depth++;
                    
                } else if(hit1) {
                    /*
                     dist2 is infinite and dist1 is finite
                     */
                    retrieved_current = true;
                    
                    branch |= (2 << depth);
                    lastIdx = currentIdx;
                    currentIdx = child1;
                    current = c1;
                    depth++;
                    
                } else if(hit2) {
                    /*
                     dist1 is infinite and dist2 is finite
                     */
                    retrieved_current = true;
                    
                    lastIdx = currentIdx;
                    currentIdx = child2;
                    current = c2;
                    depth++;
                    
                } else {
                    /*
                     Both distances are infinite
                     */
                    goingUp = true;
                }
            } else {
                
                for (int i = child1; i < child2; i++)  {
                    
                    if(intersect_triangle(&triangles[i], ray, point, normal, t, false)) {
                        *triangle_id = i;
                        *sphere_id = -1;
                    }
                    
                }
                
                goingUp = true;
            }
            continue;
        }
        
        swapped = (2 << depth) & branch;
        child1_mod = swapped ? child2 : child1;
        child2_mod = swapped ? child1 : child2;
        
        if(lastIdx == child2_mod) {
            /*
             If done parsing right node, done with entire branch
             */
            goingUp = true;
            
        } else if(lastIdx == child1_mod) {
            lastIdx = currentIdx;
            currentIdx = child2_mod;
            depth++;
        }
    }
}

bool intersect_scene(__global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes,
                     __constant Material* materials, const Ray* ray, float3* point, float3* normal, float* t,
                     Material* m, const unsigned int sphere_count, const unsigned int triangle_count,
                     const unsigned int node_count, const unsigned int material_count) {
    
    *t = INF;
    
    intersect_ground(ray, point, normal, t);
    
    int sphere_id = -1;
    
    for (unsigned int i = 0; i < sphere_count; i++)  {
        
        if(intersect_sphere(&spheres[i], ray, point, normal, t)) {
            sphere_id = i;
        }
    }
    
    int triangle_id = -1;
    
    if(node_count > 0)
        intersect_bvh(triangles, nodes, ray, point, normal, t, triangle_count, node_count, &triangle_id, &sphere_id);
    
    if(sphere_id != -1) {
        int i = sphere_id;
        *point = ray->origin + (*t)*ray->dir;
        *normal = native_divide(*point - spheres[i].pos, spheres[i].radius);
        *m = materials[spheres[i].mtlidx];
    } else if(triangle_id != -1) {
        int i = triangle_id;
        *normal = normalize(cross(*point, *normal));
        *point = ray->origin + (*t)*ray->dir;
        *m = materials[triangles[i].mtlidx];
    }
    
    return *t < INF;
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
    *wr = cosineRandHemi(n, u1, u2);
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

void dielectric_brdf(unsigned int *seed, float3 n, float3 wo, float3* kd, float3* wr,
                     float* brdf, float ior, float roughness) {
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
        
        float3 one = (float3)(1.0f, 1.0f, 1.0f);
        *kd = one;
        
        float idm = dot(i, m);
        *wr = normalize(2.0f*idm*m - i);
        
        *brdf = fabs(native_divide(idm, dot(i, n)*ct)) * GGX_G(i, *wr, m, n, a_g);
    }
    else {
        
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
          Material m, float* brdf) {
    
    int type = m.type;
    if(type == 0) {
        diffuse_brdf(seed, n, wr);
    }
    else if(type == 1) {
        plastic_brdf(seed, n, wo, kd, wr, brdf, m.roughness);
    }
    else if(type == 2) {
        dielectric_brdf(seed, n, wo, kd, wr, brdf, m.ior, m.roughness);
    }
    else if(type == 3) {
        mirror_brdf(n, wo, wr);
    }
    
}


float3 trace(__global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes,
             __constant Material* materials, const Ray* camray, const unsigned int sphere_count,
             const unsigned int triangle_count, const unsigned int node_count,
             const unsigned int material_count, unsigned int *seed, int ibl_width, int ibl_height,
             __global float3* ibl) {
    
    Ray ray = *camray;
    
    float3 throughput = (float3)(1.0f, 1.0f, 1.0f);
    float3 color = (float3)(0.0f, 0.0f, 0.0f);
    
    float3 point;
    float3 normal;
    float t;
    float brdf;
    Material mtl;
    float3 wr;
    
    for(int n = 0; n < 50; n++) {
        
        mtl = ground;
        brdf = 1.0f;
        
        ray.inv_dir = native_recip(ray.dir);
        
        if(intersect_scene(spheres, triangles, nodes, materials, &ray, &point, &normal, &t, &mtl,
                           sphere_count, triangle_count, node_count, material_count)) {
            float3 kd = mtl.kd;
            float3 ke = mtl.ke;
            bsdf(seed, normal, -1.0f * ray.dir, &wr, &kd, &ke, mtl, &brdf);
            ray.dir = wr;
            ray.origin = point;
            color += throughput * ke;
            throughput *= kd * brdf;
            
            if(n > 3) {
                float p = max(throughput.x, max(throughput.y, throughput.z));
                if(rand(seed) > p)
                    return color;
                
                throughput *= native_recip(p);
            }
        }
        else {
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
            /*return (color + throughput * void_color * ibl_sample);*/
            const float3 void_color = (float3) (0.05f, 0.05f, 0.1f);
            return (color + throughput * void_color * ibl_sample);
            /*const float3 void_color = (float3) (0.05f, 0.05f, 0.1f);
             return color += throughput * void_color;*/
            
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
                            __global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes,
                            __constant Material* materials, __constant const Camera* cam, int framenumber) {
    
    const unsigned int width = usefulnums[0];
    const unsigned int height = usefulnums[1];
    const unsigned int ibl_width = usefulnums[2];
    const unsigned int ibl_height = usefulnums[3];
    const unsigned int sphere_amt = usefulnums[4];
    const unsigned int triangle_amt = usefulnums[5];
    const unsigned int node_amt = usefulnums[6];
    const unsigned int material_amt = usefulnums[7];
    const unsigned int samples = usefulnums[8];
    const int wii = get_global_id(0);
    const int work_item_id = pix_coord(width, height, wii);
    unsigned int seed = randoms[work_item_id];
    unsigned int x_coord = work_item_id % width;    /* x-coordinate of the pixel */
    unsigned int y_coord = work_item_id / width;    /* y-coordinate of the pixel */
    
    float3 result = (float3)(0.0f, 0.0f, 0.0f);
    int spp = framenumber + 1;
    
    Ray camray = createCamRay(x_coord, height - y_coord, width, height, &seed, cam);
    /*float u = (float)x_coord / width;
     float v = (float)y_coord / height;
     result += (float3)(u, v, 1 - u - v);*/
    float3 currres = trace(spheres, triangles, nodes, materials, &camray,
                           sphere_amt, triangle_amt, node_amt, material_amt,
                           &seed, ibl_width, ibl_height, ibl);
    
    if((isnan(currres.x) != 1) && (isnan(currres.y) != 1) && (isnan(currres.z) != 1))
        result = currres;
    
    float ispp = native_recip((float) spp);
    accumbuffer[work_item_id] *= 1.0f - ispp;
    accumbuffer[work_item_id] += result * ispp;
    randoms[work_item_id] = seed;
    float3 res = tonemapFilmic(accumbuffer[work_item_id]);
    
    union Color fcolor;
    fcolor.components = (uchar4)(convert_uchar3(clamp(res, 0.0f, 1.0f) * 255), 1);
    
    output[work_item_id] = (float3)(x_coord, y_coord, fcolor.c);
}
