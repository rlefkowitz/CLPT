__constant const float eps = 3e-5f;
__constant const float INF = 1e20f;
__constant const float PI = 3.14159265358979324f;
__constant const float invPI = 0.31830988618379067f;
__constant const bool bilerp = false;
__constant const float4 csba = (float4)(0.30901699437494742f, -0.951056516295153572f, 0.951056516295153572f, 0.30901699437494742f);
__constant const float3 ZENITH_DIR = (float3)(1000.0f, 500.0f, -500.0f);
__constant const float3 NORMALIZED_ZENITH_DIR = (float3)(0.81649658f, 0.40824829f, -0.40824829f);

typedef struct RayLw {
    float3 origin;
    float3 dir;
} RayLw;


typedef struct Chunk {
    int i;
    int f;
    int dummy0;
    int dummy1;
} Chunk;

typedef struct Ray {
    float3 origin;
    float3 dir;
    float3 inv_dir;
} Ray;

typedef struct BVHNode {
    float8 box;
    int parent, child1, child2, isLeaf;
    int dummy[4];
} BVHNode;

typedef struct Medium {
    float3 absCoefficient;
    /*float3 absCoefficientEmit;*/
    float scatterCoefficient;
    int dummy[3];
} Medium;

typedef struct Material {
    float3 kd;
    float3 ke;
    float roughness;
    float ior;
    int type;
    int medIdx;
    int kdtex;
    int ketex;
    int d_tex;
    float d;
} Material;

__constant Material ground = {(float3)(0.1f, 0.1f, 0.1f), (float3)(0.0f, 0.0f, 0.0f), 0.7f, 1.0f, 5, -1, -1};

typedef struct TextureData {
  int w; /* width */
  int h; /* height */
  int s; /* start position in atlas */
  int dummy;
} TextureData;

typedef struct Sphere {
    float4 pr;
    int mtlidx;
    int dummy[3];
} Sphere;

typedef struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    int mtlidx;
    int dummy[3];
} Triangle;

typedef struct TriangleData {
  float4 t0;
  float4 t1;
  float4 t2;
  float4 n0;
  float4 n1;
  float4 n2;
  float3 uv;
  float3 vv;
} TriangleData;

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

inline float3 tonemapFilmic(const float3 f) {
    float3 x = max(0.0f, f - 0.004f);
    float3 xm = 6.2f*x;
    return native_divide(x*(xm + 0.5f), x*(xm + 1.7f) + 0.06f);
}

float2 samplePoly(unsigned int *seed) {
    int triIdx = (int) (rand(seed) * 5.0f);

    float r1 = native_sqrt(rand(seed));

    float r2 = rand(seed);
    float t = triIdx*1.2566370614359173f;
    float2 a0 = r1 * native_cos((float2)(1.57079632679489662f + t, t));
    float4 a0ex = csba * a0.xyxy;
    float2 a1 = a0 - a0ex.xz - a0ex.yw;
    return (a0 - r2 * a1);
}


Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height, const bool use_DOF,
                 unsigned int *seed, __constant const Camera* cam) {

    float offx = rand(seed);
    float offy = rand(seed);

    float x0 = x_coord + offx - 0.5f - (width >> 1);
    float y0 = (height >> 1) - y_coord - offy + 0.5f;

    const float3 fd = cam->fd;
    const float3 up = cam->up;
    const float3 rt = cross(up, fd);
    const float3 raydir = native_divide(x0*rt + y0*up, (float) height) + fd;

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

RayLw createCamRayLw(const int x_coord, const int y_coord, const int width, const int height, const bool use_DOF,
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
    RayLw ray = {cam->pos, normalize(raydir)};

    if(use_DOF) {
        float2 pt = cam->aperture_radius*samplePoly(seed);
        float3 aptOffset = up*pt.y + rt*pt.x;
        float3 dn = normalize(raydir*cam->focal_distance - aptOffset);
        ray.origin += aptOffset;
        ray.dir = dn;
    }
    return ray;
}

inline float3 orientTo(float3 a, float3 N) {
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

void buildOrthonormalBasis(float3* omega_1, float3* omega_2, float3 omega_3) {
  if(omega_3.y < -0.999999f) {
    *omega_1 = (float3){0, 0, 1};
    *omega_2 = (float3){1, 0, 0};
  }
  else {
    float a = native_recip(1.0f + omega_3.y);
    float b = -omega_3.x * omega_3.z * a;
    *omega_1 = (float3){1.0f - omega_3.x * omega_3.x * a, -omega_3.x, b};
    *omega_2 = (float3){b, -omega_3.z, 1.0f - omega_3.z * omega_3.z * a};
  }
}

bool intersect_sphere(__global Sphere* sphere, const Ray* ray, float3* point, float3* normal, float* t) {
    float3 rayToCenter = sphere->pr.xyz - ray->origin;
    float b = dot(rayToCenter, ray->dir);
    float c = dot(rayToCenter, rayToCenter) - sphere->pr.w*sphere->pr.w;
    float d = b * b - c;

    if (d <= eps) return false;
    /* must be positive */
    d = native_sqrt(d);
    /* closer potential intersection */
    float temp = b - d;
    /* if closer potential intersection is further than furthest acceptable distance, reject */
    if(temp > *t) return false;

    /* if it's closer than would be allowed (suppose self-intersection) */
    if (temp < eps) {
        /* further potential intersection */
        temp = b + d;
        /* if further potential intersection is not in acceptable distance range, reject */
        if (temp < eps || temp > *t) return false;
    }

    *t = temp;
    return true;
}

inline bool intersect_triangle(__global Triangle *triangle, const Ray* ray, float3* point, float3* normal, float* t) {

    float3 pvec = cross(ray->dir, triangle->v2);
    float invDet = native_recip(dot(triangle->v1, pvec));

    float3 tvec = ray->origin - triangle->v0;
    float u = dot(tvec, pvec) * invDet;
    if(u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, triangle->v1);
    float v = dot(ray->dir, qvec) * invDet;
    if(v < 0 || u + v > 1) return false;

    float temp = dot(triangle->v2, qvec) * invDet;
    if(temp < eps || temp > *t) return false;

    *point = triangle->v1;
    *normal = triangle->v2;
    *t = temp;
    return true;
}

__constant const float3 GROUND_NORM = (float3)(0.0f, 1.0f, 0.0f);

inline bool intersect_ground(const Ray* ray, float3* point, float3* normal, float* t) {
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

inline bool intersect_aabb(__global BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box.lo.xyz - r->origin) * invD;
    float3 t1s = (b->box.hi.xyz - r->origin) * invD;

    float3 tsmaller = fmin(t0s, t1s);
    float3 tbigger = fmax(t0s, t1s);

    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    float tmax = fmin(*t, fmin(tbigger[0], fmin(tbigger[1], tbigger[2])));

    return (tmin <= tmax);
}

bool intersect_aabb_lw(__global BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box.lo.xyz - r->origin) * invD;
    float3 t1s = (b->box.hi.xyz - r->origin) * invD;

    float3 tsmaller = fmin(t0s, t1s);

    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));

    return (tmin <= *t);
}

inline float intersect_aabb_dist(__global BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box.lo.xyz - r->origin) * invD;
    float3 t1s = (b->box.hi.xyz - r->origin) * invD;

    float3 tsmaller = fmin(t0s, t1s);
    float3 tbigger = fmax(t0s, t1s);

    float tmin = fmax(eps, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    float tmax = fmin(*t, fmin(tbigger[0], fmin(tbigger[1], tbigger[2])));

    return (tmin <= tmax) ? tmin : INF;
}

inline float4 hit_simul(__global BVHNode* b0, __global BVHNode* b1, float16 invD, float16 origin, float* t) {
    float16 a = (float16)(b0->box, b1->box);
    float16 b = (a.s012389AB4567CDEF - origin) * invD;

    float16 c = (float16)(fmin(b.lo, b.hi), fmax(b.lo, b.hi));
    c.s37 = eps;
    c.sBF = *t;

    float4 d0 = fmax(c.lo.even, c.lo.odd);
    float4 d1 = fmin(c.hi.even, c.hi.odd);

    return (float4) (fmax(d0.even, d0.odd), fmin(d1.even, d1.odd));
}

void intersect_bvh(__global Triangle* triangles, __global BVHNode* nodes, const Ray* ray, float3* point,
                   float3* normal, float* t, int* triangle_id, int* sphere_id) {

    /* TODO: get rid of major divergence at this point */
    if(!intersect_aabb(&nodes[0], ray, t)) 
      return;

    BVHNode current = nodes[0];

    float16 invD = ray->inv_dir.xyzxxyzxxyzxxyzx;
    float16 origin = ray->origin.xyzxxyzxxyzxxyzx;

    float16 b;
    int currentIdx = 0;       /* index of current node */
    int lastIdx = -1;         /* index of the previous node (for backtrace) */
    unsigned int depth = 1;   /* current depth */
    unsigned int branch = 0;  /* TODO: What was this???? Update: I think I was smarter than I think, I think it keeps track of which branch for depths 1-32 */
    bool goingUp = false;     /* whether currently backtracing */
    bool swapped;             /* if L & R were swapped */
    int child1;               /* index of child 1 */
    int child2;               /* index of child 2 */
    int child1_mod;           /* TODO: what was this */
    int child2_mod;           /* TODO: what was this */
    float dist1;
    float dist2;
    float hit1;
    float hit2;
    bool swapping;
    bool reverse;
    int i;

    /* main loop */
    /* TODO: reduce execution divergence */
    while(true) {
        
        /*
          If backtracing, work up the tree, else keep going down
         */
        if(goingUp) {
            if(currentIdx == 0) /* If traced to the top, stop intersection */
                return;

            /* set last index to previous, and current to parent (working up) */
            lastIdx = currentIdx;
            currentIdx = current.parent;
            branch &= (1 << depth) - 1;   /* TODO: what is this fuckery */
            depth--;
        } else {
            depth++;
        }

        current = nodes[currentIdx];

        /*
         BVH Traversal: Either of the first two conditions are true if currently moving up the tree
         */
        child1 = current.child1;
        child2 = current.child2;

        goingUp = false;

        if(currentIdx < lastIdx) {
            swapped = (1 << depth) & branch;        /* whether child1 should be traversed last */
            child1_mod = swapped ? child2 : child1;
            child2_mod = swapped ? child1 : child2;
            
            goingUp = lastIdx == child2_mod;
            lastIdx = goingUp ? lastIdx : currentIdx;
            currentIdx = goingUp ? currentIdx : child2_mod;
        } else if(current.isLeaf != 1) {
            /* intersect both simultaneously */
            b = (float16)(nodes[child1].box, nodes[child2].box);
            /* restructure b: (c1.box[0], c1.box[1], c2.box[0], c2.box[1]) 
                                -> (c1.box[0], c2.box[0], c1.box[1], c2.box[1]) */
            b = (b.s012389AB4567CDEF - origin) * invD;

            b = (float16)(fmin(b.lo, b.hi), fmax(b.lo, b.hi));

            /* insert constants to compare with */
            b.s37 = eps;
            b.sBF = *t;

            /* run vectorized min/max so b.s0123 = (tmin0, tmin1, tmax0, tmax1) */
            b.s0123 = fmax(b.s0246, b.s1357);
            b.s4567 = fmin(b.s8ACE, b.s9BDF);
            b.s01 = fmax(b.s02, b.s13);
            b.s23 = fmin(b.s46, b.s57);
            
            hit1 = b.s0 <= b.s2;
            hit2 = b.s1 <= b.s3;
            dist1 = hit1 ? b.s0 : INF;
            dist2 = hit2 ? b.s1 : INF;

            goingUp = !hit1 && !hit2; /* hits neither, go back up the tree */
            if(!goingUp) {
                lastIdx = currentIdx;

                reverse = dist2 < dist1;
                swapping = hit1 && (reverse || !hit2);  /* whether child1 should be traversed last */
                branch |= swapping << depth;  /* store swapping at the bit corresponding to depth in branch */
                currentIdx = reverse ? child2 : child1; /* next child will be the closest (or only) child hit */
            }
        } else {

            for (i = child1; i < child2; i++)  {

                if(intersect_triangle(&triangles[i], ray, point, normal, t)) {
                    *triangle_id = i;
                    *sphere_id = -1;
                }

            }

            goingUp = true;
        }
    }
}

bool intersect_scene(__global Sphere* spheres, __global Triangle* triangles, 
                     __global TriangleData* triangleData, 
                     __global BVHNode* nodes, const Ray* ray, float3* point, 
                     float3* normal, float* t, int* midx, float2* uv, 
                     const unsigned int sphere_count, 
                     const unsigned int node_count, const bool use_ground) {

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
        intersect_bvh(triangles, nodes, ray, point, normal, t, &triangle_id, &sphere_id);

    if(sphere_id != -1) {
        int i = sphere_id;
        *point = ray->origin + (*t)*ray->dir;
        *normal = native_divide(*point - spheres[i].pr.xyz, spheres[i].pr.w);
        *midx = spheres[i].mtlidx;
    } else if(triangle_id != -1) {
        int i = triangle_id;
        *normal = normalize(cross(*point, *normal));
        *point = ray->origin + (*t)*ray->dir;
        *midx = triangles[i].mtlidx;
        *uv = (float2) {fabs(dot(triangleData[i].uv, *point)), fabs(dot(triangleData[i].vv, *point))};
        if (triangleData[i].t0.w != 0) {
            *uv = (uv->x * triangleData[i].t0 + uv->y * triangleData[i].t1 + (1.0f - uv->x - uv->y) * triangleData[i].t2).xy;
            *uv = (float2){uv->x, 1.0f - uv->y};
        }
    } else {
        *uv = ((*point).xz + 40.0f) / 80.0f;
    }

    return *t < ti;
}

float4 bilerp_func(float4 a, float4 b, float4 c, float4 d, float x, float y) {
    float g = 1.0f - x, h = 1.0f - y;
    float i = g*h, j = x*h, k = g*y, l = x*y;
    return a*i + b*j + c*k + d*l;
}

float4 sampleImage(float u, float v, int width, int height, __global float4* img, int offset) {
    if(bilerp) {
        float px = width * u - 0.5f;
        float py = height * v - 0.5f;
        int p0x = (int) px;
        int p0y = (int) py;
        int p1x = (p0x + 1) % width;
        int p2y = (p0y + 1) % height;
        float4 s0 = img[offset + p0y*width + p0x];
        float4 s1 = img[offset + p0y*width + p1x];
        float4 s2 = img[offset + p2y*width + p0x];
        float4 s3 = img[offset + p2y*width + p1x];
        return bilerp_func(s0, s1, s2, s3, px - p0x, py - p0y);
    }
    int x = (int) (u * width);
    int y = (int) (v * height);
    return img[offset + y*width + x];
}

/*

  MS Utility functions

 */
float MS_sign(float a) {
    return a < 0 ? -1.0f : 1.0f;
}

float MS_Lambda(float3 w, float a_g) {
    if(w.y > 0.9999f) return 0.0f;
    if(w.y < -0.9999f) return -1.0f;

    float a = a_g * a_g * (native_recip(w.y * w.y) - 1.0f);

    return 0.5f * (MS_sign(w.y) * native_sqrt(1.0f + a) - 1.0f);
}

float MS_C1(float h) {
    return fmin(1.0f, fmax(0.0f, 0.5f * h + 0.5f));
}

float MS_invC1(float U) {
    return fmax(-1.0f, fmin(1.0f, 2.0f * U - 1.0f));
}

float MS_G1dist(float3 wi, float h, float a_g) {
    if(wi.y > 0.9999f) return 1.0f;
    if(wi.y <= 0.0f) return 0.0f;
    return native_powr(MS_C1(h), MS_Lambda(wi, a_g));
}

float MS_sampleHeight(float3 wr, float hr, float u, float a_g) {
    if(wr.y > 0.9999f) 
        return MAXFLOAT;
    if(wr.y < -0.9999f) 
        return MS_invC1(u*MS_C1(hr));
    if(fabs(wr.y) < 0.0001f) 
        return hr;

    float G1 = MS_G1dist(wr, hr, a_g);

    if (u > 1.0f - G1) return MAXFLOAT;
    
    return MS_invC1(native_divide(MS_C1(hr), native_powr(1.0f - u, native_recip(MS_Lambda(wr, a_g)))));
}

/*

  MSCond BSDF functions

 */
float MSCond_fresnel(float n, float k, float c) {
    float n2pk2 = n * n + k * k;
    float ncd = 2.0f * n * c;
    float c2 = c * c;
    float rs_num = n2pk2 - ncd + c2;
    float rs_den = n2pk2 + ncd + c2;
    float rs = native_divide(rs_num, rs_den);

    float rp_num = n2pk2 * c2 - ncd + 1;
    float rp_den = n2pk2 * c2 + ncd + 1;
    float rp = native_divide(rp_num, rp_den);
    return 0.5f * (rs + rp);
}

float3 MSCond_color(float3 n, float3 k, float c) {
    float c2 = c*c;
    float3 n2pk2 = n * n + k * k;
    float3 n_2 = 2.0f * n;
    float3 ncd = n_2 * c;
    float3 ncd_2 = 2.0f*ncd;
    float3 rs = 2.0f - native_divide(ncd_2, n2pk2 + ncd + c2)
                     - native_divide(ncd_2, (n2pk2)*c2 + ncd + 1.0f);
    float3 refl = 0.5f - native_divide(n_2, n2pk2 + n_2 + 1.0f);

    return (refl*rs);
}

float3 MSCond_sampleGGXVNDF(float3 wo, float u1, float u2, float a_g) {
    float3 v = normalize((float3){wo.x * a_g, wo.y, wo.z * a_g});

    float3 t1 = (v.y < 0.9999f) ? normalize(cross(v, (float3){0, 1, 0})) : (float3){1, 0, 0};
    float3 t2 = cross(t1, v);

    float a = native_recip(1.0f + v.y);
    float r0 = native_sqrt(u1);
    float phi = (u2  <  a) ? native_divide(u2, a) * M_PI : M_PI + M_PI * native_divide(u2 - a, 1.0f - a);
    float p1 = r0 * native_cos(phi);
    float p2 = r0 * native_sin(phi) * ((u2 < a) ? 1.0f : v.y);

    float3 n = t1 * p1 + t2 * p2 + v * native_sqrt(fmax(0.0f, 1.0f - p1 * p1 - p2 * p2));

    return normalize((float3){a_g * n.x, fmax(0.0f, n.y), a_g * n.z});
}


void MSCond_sampleCond(unsigned int *seed, float3 *wo, float3 *w, float3 n, float3 k, float a_g) {
    float3 _wo = -(*wo);
    float3 wm = MSCond_sampleGGXVNDF(_wo, rand(seed), rand(seed), a_g);

    float cosTheta = dot(_wo, wm);
    float absCosTheta = fabs(cosTheta);
    float3 albedo = MSCond_color(n, k, absCosTheta);

    *wo += wm * 2.0f * cosTheta;
    *w *= albedo;
}


void MSCond_bsdf(unsigned int *seed, float3 n, float3 wo, float3* kd, float3 *ke ,float3* wr,
                     float* brdf, float3 np, float3 kp, float roughness) {
    float h = 1.0f;
    float div = native_rsqrt(fabs(n.z) > 0 ? n.z * n.z + n.x * n.x : n.y * n.y + n.x * n.x);
    float3 u = div * (fabs(n.z) > 0 ? (float3){-n.z, 0, n.x} : (float3){n.y, -n.x, 0});
    float3 v = cross(n, u);

    float3 _wo = -wo.x * (float3){u.x, n.x, v.x} - wo.y * (float3){u.y, n.y, v.y} - wo.z * (float3){u.z, n.z, v.z};
    float3 w = (float3){1, 1, 1};
    int r = 0;
    while (r < 1e3) {
        h = MS_sampleHeight(_wo, h, rand(seed), roughness);
        if(h == MAXFLOAT) 
            break;
        else 
            r++;
        MSCond_sampleCond(seed, &_wo, &w, np, kp, roughness);
        if((h != h) || (_wo.y != _wo.y)) {
            _wo = (float3){0, 1, 0};
            break;
        }
    }
    *wr = _wo.x * u + _wo.y * n + _wo.z * v;
    *kd = w;
    *ke = (float3)(0.0f, 0.0f, 0.0f);
    *brdf = 1.0f;
}

/*

  MSDielectric BSDF functions

 */
float MSDielectric_fresnel(float n, float k, float c) {
    float n2pk2 = n * n + k * k;
    float ncd = 2.0f * n * c;
    float c2 = c * c;
    float rs_num = n2pk2 - ncd + c2;
    float rs_den = n2pk2 + ncd + c2;
    float rs = native_divide(rs_num, rs_den);

    float rp_num = n2pk2 * c2 - ncd + 1;
    float rp_den = n2pk2 * c2 + ncd + 1;
    float rp = native_divide(rp_num, rp_den);
    return 0.5f * (rs + rp);
}

float3 MSDielectric_refract(float3 wi, float3 wm, float eta) {
    float cos_theta_i = dot(wi, wm);
    float cos_theta_t2 = 1.0f - native_divide(1.0f - cos_theta_i * cos_theta_i, eta * eta);
    float cos_theta_t = -native_sqrt(fmax(0.0f, cos_theta_t2));

    return wm * (native_divide(cos_theta_i, eta) + cos_theta_t) - native_divide(wi, eta);
}

float MSDielectric_Fresnel(float3 wi, float3 wm, float eta) {
    float cos_theta_i = dot(wi, wm);
    float cos_theta_t2 = 1.0f - native_divide(1.0f - cos_theta_i * cos_theta_i, eta * eta);

    if (cos_theta_t2 <= 0) return 1.0f;

    float cos_theta_t = native_sqrt(cos_theta_t2);

    float Rs = native_divide(cos_theta_i - eta * cos_theta_t, cos_theta_i + eta * cos_theta_t);
    float Rp = native_divide(eta * cos_theta_i - cos_theta_t, eta * cos_theta_i + cos_theta_t);

    return 0.5f * (Rs * Rs + Rp * Rp);
}

float3 MSDielectric_sampleGGXVNDF(float3 wo, float u1, float u2, float a_g) {
    float3 Vh = normalize((float3){a_g * wo.x, wo.y, a_g * wo.z});

    float lensq = Vh.x * Vh.x + Vh.z * Vh.z;
    float3 T1 = lensq > 0 ? native_rsqrt(lensq) * (float3){-Vh.z, 0, Vh.x} : (float3){1, 0, 0};
    float3 T2 = cross(Vh, T1);

    float r = native_sqrt(u1);
    float phi = 2.0f * M_PI * u2;
    float t1 = r * native_cos(phi);
    float t2 = r * native_sin(phi);
    float s = 0.5f + 0.5f * Vh.y;
    t2 = (1.0f - s) * native_sqrt(1.0f - t1 * t1) + s * t2;

    float3 Nh = T1 * t1 + T2 * t2 + Vh * native_sqrt(fmax(0.0f, 1.0f - t1 * t1 - t2 * t2));

    return normalize((float3){a_g * Nh.x, fmax(0.0f, Nh.y), a_g * Nh.z});
}

bool MSDielectric_samplePhaseFunction(unsigned int *seed, float3 *wo, float3 *w, float3 kd, bool wi_outside, float a_g, float eta) {
    float U1 = rand(seed), U2 = rand(seed);

    float3 _wo = -(*wo);

    float3 wm = wi_outside ? MSDielectric_sampleGGXVNDF(_wo, U1, U2, a_g) : -MSDielectric_sampleGGXVNDF(-_wo, U1, U2, a_g);

    float F = MSDielectric_Fresnel(-(*wo), wm, eta);

    if(rand(seed) < F) {
        *wo = wm * 2.0f * dot(_wo, wm) - _wo;
        return wi_outside;
    }
    *wo = normalize(MSDielectric_refract(_wo, wm, eta));
    *w = (*w) * kd;
    return !wi_outside;
}


void MSDielectric_bsdf(unsigned int *seed, float3 n, float3 *wr, float *brdf, float3 *kd, float3 wo, float roughness, float ior, bool* transmitted) {
    float h = 1.0f + MS_invC1(0.999f);
    float div = native_rsqrt(fabs(n.z) > 0 ? n.z * n.z + n.x * n.x : n.y * n.y + n.x * n.x);
    float3 u = div * (fabs(n.z) > 0 ? (float3){-n.z, 0, n.x} : (float3){n.y, -n.x, 0});
    float3 v = cross(n, u);

    
    float3 _wo = -wo.x * (float3){u.x, n.x, v.x} - wo.y * (float3){u.y, n.y, v.y} - wo.z * (float3){u.z, n.z, v.z};
    float3 w = (float3){1, 1, 1};
    int r = 0;
    bool outsidefirst = _wo.y > 0;
    bool outside = true;
    float eta = ior;
    while (r < 1e4) {
        float U = rand(seed);
        h = outside ? MS_sampleHeight(_wo, h, U, roughness) : -MS_sampleHeight(-_wo, -h, U, roughness);
        if(h == MAXFLOAT || h == -MAXFLOAT) break;
        else r++;

        outside = MSDielectric_samplePhaseFunction(seed, &_wo, &w, *kd, outside, roughness, outside ? eta : 1.0f / eta);
        if((h != h) || (_wo.y != _wo.y)) {
            _wo = (float3){0, 1, 0};
            break;
        }
    }

    if((_wo.y > 0) == outsidefirst) *transmitted = true;
    *wr = _wo.x * u + _wo.y * n + _wo.z * v;
    *kd = w;
    *brdf = 1.0f;
}

/*

  MSDiff BSDF functions

 */

void MSDiff_sampleDiff(unsigned int *seed, float3* wo, float3* w, float3 kd, float a_g) {
    float3 v = -normalize((float3){wo->x * a_g, wo->y, wo->z * a_g});
    
    float3 t1 = (v.y < 0.9999f) ? normalize(cross(v, (float3){0, 1, 0})) : (float3){1, 0, 0};
    float3 t2 = cross(t1, v);

    float u1 = rand(seed), u2 = rand(seed);

    float a = native_recip(1.0f + v.y);
    float r = native_sqrt(u1);
    float phi = (u2 < a) ? native_divide(u2, a) * M_PI : M_PI + M_PI * native_divide(u2 - a, 1.0f - a);
    float p1 = r * native_cos(phi);
    float p2 = r * native_sin(phi) * ((u2 < a) ? 1.0f : v.y);

    float3 n = t1 * p1 + t2 * p2 + v * native_sqrt(fmax(0.0f, 1.0f - p1 * p1 - p2 * p2));

    float3 wm = normalize((float3){a_g * n.x, fmax(0.0f, n.y), a_g * n.z});
    float3 w1, w2;
    buildOrthonormalBasis(&w1, &w2, wm);

    float r1 = 2.0f * rand(seed) - 1.0f;
    float r2 = 2.0f * rand(seed) - 1.0f;

    float phi_ = 0.0f, r_ = 0.0f;
    if (r1 * r1 > r2 * r2) {
        r_ = r1;
        phi_ = 0.25f * M_PI * native_divide(r2, r1);
    } else if (r1 != 0 || r2 != 0) {
        r_ = r2;
        phi_ = M_PI * (0.5f - 0.25f * native_divide(r1, r2));
    }
    float x = r_ * native_cos(phi_);
    float y = r_ * native_sin(phi_);
    float z = native_sqrt(fmax(0.0f, 1.0f - x * x - y * y));
    *wo = w1 * x + w2 * y + wm * z;
    *w = (*w) * kd;
}

void MSDiff_bsdf(unsigned int *seed, float3 n, float3 wo, float3 *kd, float3 *wr, float *brdf, float roughness) {
  float h = 1.0f;
    float div = native_rsqrt(fabs(n.z) > 0 ? n.z * n.z + n.x * n.x : n.y * n.y + n.x * n.x);
    float3 u = div * (fabs(n.z) > 0 ? (float3){-n.z, 0, n.x} : (float3){n.y, -n.x, 0});
    float3 v = cross(n, u);


    float3 _wo = -wo.x * (float3){u.x, n.x, v.x} - wo.y * (float3){u.y, n.y, v.y} - wo.z * (float3){u.z, n.z, v.z};
    float3 w = (float3){1, 1, 1};
    int r = 0;
    while (r < 1e3) {
        h = MS_sampleHeight(_wo, h, rand(seed), roughness);
        if(h == MAXFLOAT) 
            break;
        else 
            r++;
        MSDiff_sampleDiff(seed, &_wo, &w, *kd, roughness);
        if((h != h) || (_wo.y != _wo.y)) {
            _wo = (float3){0, 1, 0};
            break;
        }
    }
    *wr = _wo.x * u + _wo.y * n + _wo.z * v;
    *kd = w;
    *brdf = 1.0f;
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

    float epsilon = min(eps, rand(seed));

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
        *wr = 2.0f*idm*m - i;
    } else {
        float u1 = rand(seed);
        float u2 = rand(seed);
        *wr = cosineRandHemi(n, u1, u2);
    }


    *brdf = fabs(native_divide(idm, dot(i, n)*ct)) * GGX_G(i, *wr, m, n, a_g);

}



/*

 Dielectric BSDF Functions

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

void dielectric_bsdf(unsigned int *seed, float3 n, float3 wo, float3* kd, float3 *ke ,float3* wr,
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

 Metal BRDF Functions

 */

float3 metal_color(float3 n, float3 k, float c) {
    float c2 = c*c;
    float3 n2pk2 = n*n + k*k;
    float3 n_2 = 2.0f*n;
    float3 ncd = n_2*c;
    float3 ncd_2 = 2.0f*ncd;
    float3 rs = 2.0f - native_divide(ncd_2, n2pk2 + ncd + c2)
                     - native_divide(ncd_2, (n2pk2)*c2 + ncd + 1.0f);
    float3 refl = 0.5f - native_divide(n_2, n2pk2 + n_2 + 1.0f);

    return (refl*rs);
}

void metal_brdf(unsigned int *seed, float3 n, float3 wo, float3* kd, float3 *ke ,float3* wr,
                     float* brdf, float3 np, float3 kp, float roughness) {
    float3 i = wo;
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

    float cosTheta = dot(i, m);
    float absCosTheta = fabs(cosTheta);
    *kd = metal_color(np, kp, absCosTheta);
    *ke = (float3)(0.0f, 0.0f, 0.0f);

    *wr = normalize(2.0f*cosTheta*m - i);

    *brdf = fabs(native_divide(cosTheta, dot(i, n)*ct)) * GGX_G(i, *wr, m, n, a_g);
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
        diffuse_brdf(seed, sign(dot(n, wo)) * n, wr);
    }
    else if(type == 1) {
        plastic_brdf(seed, n, wo, kd, wr, brdf, m.roughness);
    }
    else if(type == 2) {
        dielectric_bsdf(seed, n, wo, kd, ke, wr, brdf, m.ior, m.roughness, transmitted);
    }
    else if(type == 3) {
        metal_brdf(seed, sign(dot(n, wo)) * n, wo, kd, ke, wr, brdf, m.kd, m.ke, m.roughness);
    }
    else if(type == 4) {
        mirror_brdf(n, wo, wr);
    }
    else if(type == 5) {
        MSDiff_bsdf(seed, sign(dot(n, wo)) * n, wo, kd, wr, brdf, m.roughness);
    }
    else if(type == 6) {
        MSDielectric_bsdf(seed, n, wr, brdf, kd, wo, m.roughness, m.ior, transmitted);
    }
    else if(type == 7) {
        MSCond_bsdf(seed, sign(dot(n, wo)) * n, wo, kd, ke, wr, brdf, m.kd, m.ke, m.roughness);
    }

}


float medSampleADist(Medium med, float maxDist, unsigned int *seed) {
    return native_divide(-native_log(rand(seed)), med.scatterCoefficient);
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
    *volPDF = 0.25 * M_1_PI;
    float theta = 2.0 * PI * rand(seed);
    float r1 = 1.0f - 2.0f * rand(seed);
    float sp = native_sqrt(1.0f - r1*r1);
    float x = sp * native_cos(theta);
    float y = sp * native_sin(theta);
    return (float3)(x, y, r1);
}

float medScatterDirPdf() {
    return 0.25 * M_1_PI;
}

float3 medTransmission(Medium med, float distance) {
    return native_exp(-med.absCoefficient * distance);
}

/*float3 medTransmissionEmit(Medium med, float distance) {
    float3 ad = -med.absCoefficientEmit * distance;
    return native_exp(ad);
}*/


__constant float airAbsDist = 1.0f;
__constant float airScatterDist = 160.0f;
__constant Medium air = {(float3)(0.0102586589f, 0.0102586589f, 0.0102586589f), 1.0f / 20.0f};


float3 trace(__global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes,
             __constant Material* materials, __constant Medium* mediums, __global TriangleData* triangleData, 
             __constant TextureData* textureData, __global float4* textureAtlas, 
             const Ray* camray, const unsigned int sphere_count, const unsigned int triangle_count, 
             const unsigned int node_count, const unsigned int material_count, const unsigned int medium_count, 
             unsigned int *seed, int ibl_width, int ibl_height, __global float4* ibl, const float3 void_color, 
             const bool use_IbL, const bool use_ground) {

    Ray ray = *camray;

    float3 throughput = (float3)(1.0f, 1.0f, 1.0f);
    float3 color = (float3)(0.0f, 0.0f, 0.0f);

    float3 point;
    float3 normal;
    float t;
    float brdf;
    float pdf;
    int mtlidx;
    Material mtl;
    float3 wr;
    bool transmitted;
    int currMedIdx = -1;
    Medium med = air;
    bool hitSurface = false;
    float volWeight = 1.0f;
    float volPDF = 1.0f;
    float maxDist;
    bool hitThisTime;
    float p;
    float d;
    float3 kd;
    float3 ke;
    float2 uv;

    for(int n = 0; n < 1500; n++) {

        mtlidx = -1;
        brdf = 1.0f;
        pdf = 1.0f;

        ray.inv_dir = native_recip(ray.dir);

        maxDist = 1e16f;

        if(currMedIdx != -2) {
            maxDist = medSampleADist(med, maxDist, seed);
        }
        t = maxDist;

        hitThisTime = intersect_scene(spheres, triangles, triangleData, nodes, 
                                      &ray, &point, &normal, &t, &mtlidx, &uv, 
                                      sphere_count, node_count, use_ground);


        mtl = mtlidx < 0 ? ground : materials[mtlidx];

        hitSurface = true;

        if(!hitThisTime) {
            float3 env_map_pos = (float3)(0.0f, 15.0f, 0.0f);
            float3 eye = ray.origin - env_map_pos;
            float b = dot(eye, ray.dir);
            const float c = dot(eye, eye) - 1e5f;
            float d = b*b - c;
            t = native_sqrt(d) - b;
            hitSurface = false;
        }

        if(n > 4) {
          p = max(throughput.x, max(throughput.y, throughput.z));
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
            ray.dir = medSampleScatterDir(&volPDF, seed);
            /*throughput *= 4.0f * M_PI_F;*/
            continue;
          }
        }

        if(hitSurface) {

            d = mtl.d;
            kd = mtl.kd;
            ke = mtl.ke;
            if (mtl.kdtex != -1) {
              kd = mtl.kd * sampleImage(uv.x, uv.y, textureData[mtl.kdtex].w, textureData[mtl.kdtex].h, textureAtlas, textureData[mtl.kdtex].s).xyz;
            }
            if (mtl.ketex != -1) {
              ke = mtl.ke * sampleImage(uv.x, uv.y, textureData[mtl.ketex].w, textureData[mtl.ketex].h, textureAtlas, textureData[mtl.ketex].s).xyz;
            }
            if (mtl.d_tex != -1) {
              d = sampleImage(uv.x, uv.y, textureData[mtl.d_tex].w, textureData[mtl.d_tex].h, textureAtlas, textureData[mtl.d_tex].s).x;
            }

            transmitted = false;

            if(rand(seed) > d) {
              wr = ray.dir;
              kd = (float){1, 1, 1};
              transmitted = true;
            } else {
              bsdf(seed, normal, -1.0f * ray.dir, &wr, &kd, &ke, mtl, &brdf, &transmitted);
            }

            color += throughput * ke;

            if(transmitted) {
              int mtlMed = mtl.medIdx;
              if(mtlMed != currMedIdx) {
                currMedIdx = mtlMed;
                med = mediums[mtlMed];
              } else {
                currMedIdx = -1;
                med = air;
              }
            }
            ray.origin = point;
            ray.dir = wr;
            throughput *= kd * brdf / pdf;
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
                const float3 ibl_sample = sampleImage(u, v, ibl_width, ibl_height, ibl, 0).xyz;
                return (color + throughput * void_color * ibl_sample);
            }
            else {
                /*float cos2theta = dot(ray.dir, NORMALIZED_ZENITH_DIR);
                float costheta = native_sqrt(0.5f * (cos2theta + 1.0f));
                costheta = pow(costheta, 500.0f);
                float mult = max(130.0f * costheta + 1.0f, 1.0f);
                return color += throughput * (mult * costheta + (1.0f - costheta));*/
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
    int bigcol = (i >> 3) / height;
    int col = (bigcol << 3) + (i & 0x7);
    int rowcomp = i - (bigcol * height << 3);
    rowcomp >>= 3;
    return (rowcomp * width + col);
}

union Color{ float c; uchar4 components; };

__kernel void render_kernel(__global float3* accumbuffer, __constant unsigned int* usefulnums,
                            __global unsigned int* randoms, __global float3* ibl,
                            __global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes,
                            __constant Material* materials, __constant Medium* mediums,
                            __constant TextureData* textureData, __global float4* textureAtlas, 
                            __global TriangleData *triangleData, const float3 void_color, 
                            __constant const Camera* cam) {

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
    unsigned int y_coord = work_item_id / width;    /* y-coordinate of the pixel */
    unsigned int x_coord = work_item_id - width * y_coord;    /* x-coordinate of the pixel */

    Ray camray = createCamRay(x_coord, height - y_coord, width, height, use_DOF, &seed, cam);

    float3 currres = trace(spheres, triangles, nodes, materials, mediums,
                           triangleData, textureData, textureAtlas, &camray, sphere_amt, triangle_amt, node_amt,
                           material_amt, medium_amt, &seed, ibl_width, ibl_height,
                           ibl, void_color, use_IbL, use_ground);
                              
    currres.x = isnan(currres.x) ? 0.0f : currres.x;
    currres.y = isnan(currres.y) ? 0.0f : currres.y;
    currres.z = isnan(currres.z) ? 0.0f : currres.z;

    accumbuffer[work_item_id] += currres;
    randoms[work_item_id] = seed;
}

__kernel void final_kernel(__global float3* output, __global float3* accumbuffer, int width, int height, 
                           const int samples) {

    const int work_item_id = get_global_id(0);

    unsigned int y_coord = work_item_id / width;    /* y-coordinate of the pixel */
    unsigned int x_coord = work_item_id - width * y_coord;    /* x-coordinate of the pixel */

    output[work_item_id] = tonemapFilmic(native_divide(accumbuffer[work_item_id], samples));
}
