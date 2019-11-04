__constant const float eps = 3e-5f;
__constant const float INF = 1e20f;
__constant const float PI = 3.14159265358979324f;
__constant const float invPI = 0.31830988618379067f;
__constant const float inv4PI = 0.07957747154594767f;
__constant const float PI2 = 6.28318530717958647f;
__constant const bool bilerp = false;
__constant const float4 csba = (float4)(0.30901699437494742f, -0.951056516295153572f, 0.951056516295153572f, 0.30901699437494742f);
__constant const float3 ZENITH_DIR = (float3)(1000.0f, 500.0f, -500.0f);
__constant const float3 NORMALIZED_ZENITH_DIR = (float3)(0.81649658f, 0.40824829f, -0.40824829f);

typedef struct RayLw {
    float3 origin;
    float3 dir;
} RayLw;

typedef struct Ray {
    float3 origin;
    float3 dir;
    float3 inv_dir;
} Ray;

typedef struct BVHNode {
    float3 box[2];
    int parent, child1, child2, isLeaf;
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
    int tex0;
    int tex1;
    int type;
    int medIdx;
    int dummy[2];
} Material;

__constant Material ground = {(float3)(0.9f, 0.908f, 0.925f), (float3)(0.0f, 0.0f, 0.0f), 0.00125f, 1.0f, -1, -1, 1};

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

    if(use_DOF) {
        float2 pt = cam->aperture_radius*samplePoly(seed);
        float3 aptOffset = up*pt.y + rt*pt.x;
        float3 dn = normalize(raydir*cam->focal_distance - aptOffset);
        RayLw ray = {cam->pos + aptOffset, dn};
        return ray;
    } else {
        RayLw ray = {cam->pos, normalize(raydir)};
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
    float3 rayToCenter = sphere->pr.xyz - ray->origin;
    float b = dot(rayToCenter, ray->dir);
    float c = dot(rayToCenter, rayToCenter) - sphere->pr.w*sphere->pr.w;
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

bool intersect_triangle(__global Triangle *triangle, const Ray* ray, float3* point, float3* normal, float* t,
                        const bool cull) {

    float3 pvec = cross(ray->dir, triangle->v2);
    float det = dot(triangle->v1, pvec);

    /* Backface culling */
    /*if(det < eps && false) return false;*/

    float invDet = native_recip(det);

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

bool intersect_aabb(__global BVHNode* b, const Ray* r, float* t) {
    float3 invD = r->inv_dir;
    float3 t0s = (b->box[0] - r->origin) * invD;
    float3 t1s = (b->box[1] - r->origin) * invD;

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

float intersect_aabb_dist(__global BVHNode* b, const Ray* r, float* t) {
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
                   float3* normal, float* t, int* triangle_id, int* sphere_id) {

    if(!intersect_aabb(&nodes[0], ray, t))
        return;

    BVHNode current = nodes[0];

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
            currentIdx = current.parent;
            branch &= (2 << depth) - 1;
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

        if(current.isLeaf != 1) {
            float dist1 = intersect_aabb_dist(&nodes[child1], ray, t);
            float dist2 = intersect_aabb_dist(&nodes[child2], ray, t);
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

                if(intersect_triangle(&triangles[i], ray, point, normal, t, false)) {
                    *triangle_id = i;
                    *sphere_id = -1;
                }

            }

            goingUp = true;
        }
    }
}

bool intersect_scene(__global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes, 
                     const Ray* ray, float3* point, float3* normal, float* t, int* midx, 
                     const unsigned int sphere_count, const unsigned int node_count, 
                     const bool use_ground) {

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
    float phi = PI2*rand(seed);
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
        diffuse_brdf(seed, n, wr);
    }
    else if(type == 1) {
        plastic_brdf(seed, n, wo, kd, wr, brdf, m.roughness);
    }
    else if(type == 2) {
        dielectric_bsdf(seed, n, wo, kd, ke, wr, brdf, m.ior, m.roughness, transmitted);
    }
    else if(type == 3) {
        metal_brdf(seed, n, wo, kd, ke, wr, brdf, m.kd, m.ke, m.roughness);
    }
    else if(type == 4) {
        mirror_brdf(n, wo, wr);
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

/*float3 medTransmissionEmit(Medium med, float distance) {
    float3 ad = -med.absCoefficientEmit * distance;
    return native_exp(ad);
}*/


__constant float airScatterDist = 4000.0f;
__constant Medium air = {(float3)(0.0f, 0.0f, 0.0f), /*(float3)(INF, INF, INF), */1.0f / 4000.0f};


float3 trace(__global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes,
             __constant Material* materials, __constant Medium* mediums, const Ray* camray,
             const unsigned int sphere_count, const unsigned int triangle_count, const unsigned int node_count,
             const unsigned int material_count, const unsigned int medium_count, unsigned int *seed,
             int ibl_width, int ibl_height, __global float3* ibl, const float3 void_color, const bool use_IbL,
             const bool use_ground) {

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
    int currMedIdx = -2;
    Medium med = air;
    bool hitSurface = false;
    float volWeight = 1.0f;
    float volPDF = 1.0f;

    for(int n = 0; n < 1500; n++) {

        mtlidx = -1;
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

        bool hitThisTime = intersect_scene(spheres, triangles, nodes, &ray, &point, &normal, &t, &mtlidx,
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

        if(n > 3) {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if(rand(seed) > p)
                break;

            throughput *= native_recip(p);
        }

        if(currMedIdx != -2) {
            volWeight = 1.0f;
            float distance = medSampleDist(med, t, volWeight, volPDF, maxDist, &pdf);
            float3 transmission = medTransmission(med, distance);
            throughput *= transmission * volWeight;
            /*float3 transmissionEmit = medTransmissionEmit(med, distance);
            color += transmissionEmit * throughput;*/

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

            bsdf(seed, normal, -1.0f * ray.dir, &wr, &kd, &ke, mtl, &brdf, &transmitted);

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
    int bigcol = (i >> 3) / height;
    int col = (bigcol << 3) + (i & 0x7);
    int rowcomp = i - (bigcol * height << 3);
    rowcomp >>= 3;
    return (rowcomp * width + col);
}

union Color{ float c; uchar4 components; };

__kernel void render_kernel(__global float3* accumbuffer, __constant unsigned int* usefulnums,
                            __global unsigned int* randoms, __global float3* ibl, __global float3* output,
                            __global Sphere* spheres, __global Triangle* triangles, __global BVHNode* nodes,
                            __constant Material* materials, __constant Medium* mediums,
                            const float3 void_color, __constant const Camera* cam, int framenumber) {

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
                           &camray, sphere_amt, triangle_amt, node_amt,
                           material_amt, medium_amt, &seed, ibl_width, ibl_height,
                           ibl, void_color, use_IbL, use_ground);

    accumbuffer[work_item_id] += currres;
    randoms[work_item_id] = seed;
    float3 res = tonemapFilmic(native_divide(accumbuffer[work_item_id], framenumber + 1));

    union Color fcolor;
    fcolor.components = (uchar4)(convert_uchar3(res * 255), 1);

    output[work_item_id] = (float3)(x_coord, y_coord, fcolor.c);
}


__kernel void init_kernel(__global Ray* camera_rays, __global float3* throughputs, __global int* actual_id,
                          volatile __global int* win, __global unsigned int* randoms, __constant const Camera* cam, 
                          int width, int height, const uchar bools) {

    const int work_item_id = get_global_id(0);
    actual_id[work_item_id] = work_item_id;
    unsigned int seed = randoms[work_item_id];

    unsigned int y_coord = work_item_id / width;    /* y-coordinate of the pixel */
    unsigned int x_coord = work_item_id - width * y_coord;    /* x-coordinate of the pixel */

    atomic_add(win, 1);
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    camera_rays[work_item_id] = createCamRay(x_coord, height - y_coord, width, height,
                                             bools & 1, &seed, cam);

    throughputs[work_item_id] = (float3)(1.0f, 1.0f, 1.0f);

    randoms[work_item_id] = seed;
}


__kernel void intersection_kernel(__global unsigned char* finished, __global float3* points, __global float3* normals, 
                                  __global int* materials, __global Ray* rays, __global Sphere* spheres, 
                                  __global Triangle* triangles, __global BVHNode* nodes, __global int* actual_id, 
                                  const int sphere_amt, const int node_amt, const uchar bools) {

    const int work_item_id = actual_id[get_global_id(0)];
    
    Ray ray = rays[work_item_id];

    float t = 1e16f;

    int mtlidx = -1;
    float3 normal;
    float3 point;

    bool hit = intersect_scene(spheres, triangles, nodes, &ray, &point, &normal, &t, &mtlidx,
                               sphere_amt, node_amt, bools & 4);

    finished[work_item_id] = !hit;
    if(hit) {
        materials[work_item_id] = mtlidx;
        points[work_item_id] = point;
        normals[work_item_id] = normal;
    }

}


/*
 Shading Kernel:
  - Needs: hit, point, normal, material index, seed, current iteration
  - Gives/Sets: finished, accumbuffer, throughput
 */

__kernel void shading_kernel(__global Ray* rays, __global unsigned char* finished, __global float3* accumbuffer, 
                             __global float3* throughputs, __global int* mtlidxs, __global float3* points, 
                             __global float3* normals, __constant Material* materials, __global float3* ibl, 
                             __global int* actual_id, __global unsigned int* randoms, const int ibl_width, 
                             const int ibl_height, const float3 void_color, const uchar bools, 
                             const unsigned int current_iteration) {

    const int work_item_id = actual_id[get_global_id(0)];
    unsigned int seed = randoms[work_item_id];

    Ray ray = rays[work_item_id];

    float3 throughput = throughputs[work_item_id];

    if(finished[work_item_id] == 0) {

        int mtlidx = mtlidxs[work_item_id];
        float3 point = points[work_item_id];
        float3 normal = normals[work_item_id];
        float brdf = 1.0f;
        float3 wr;

        Material mtl = mtlidx < 0 ? ground : materials[mtlidx];

        if(current_iteration > 3) {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if(rand(seed) > p) {
                finished[work_item_id] = true;
                return;
            }

            throughput *= native_recip(p);
        }

        float3 kd = mtl.kd;
        float3 ke = mtl.ke;

        int thold;

        bsdf(seed, normal, -1.0f * ray.dir, &wr, &kd, &ke, mtl, &brdf, &thold);

        accumbuffer[work_item_id] += throughput * ke;

        ray.origin = point;
        ray.dir = wr;
        throughput *= kd * brdf;
    } else {
        if(bools & 2) {
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
            accumbuffer[work_item_id] += throughput * void_color * ibl_sample;
        } else {
            accumbuffer[work_item_id] += throughput * void_color;
        }
        return;
    }

    throughputs[work_item_id] = throughput;

    rays[work_item_id] = ray;

    randoms[work_item_id] = seed;
}


__kernel void rm_kernel(__global int* actual_id, __global unsigned char* finished, volatile  __global int *win) {

    const int work_item_id = get_global_id(0);
    const int global_size = get_global_size(0);

    int untilusage = work_item_id;

    int i = 0;
    int idx = actual_id[0];

    while(idx < global_size) {
        if(finished[idx] != 0 && --untilusage < 0) break;
        idx = actual_id[++i];
    }

    /*while((finished[idx = actual_id[++i]] != 0 || --untilusage != 0) && i < global_size);*/

    actual_id[work_item_id] = idx;

    if(i > global_size) {
        atomic_sub(win, 1);
        mem_fence(CLK_GLOBAL_MEM_FENCE);
    }
}


__kernel void final_kernel(__global float3* output, __global float3* accumbuffer, int width, int height, 
                           const int framenumber) {

    const int work_item_id = get_global_id(0);

    unsigned int y_coord = work_item_id / width;    /* y-coordinate of the pixel */
    unsigned int x_coord = work_item_id - width * y_coord;    /* x-coordinate of the pixel */

    float3 res = tonemapFilmic(native_divide(accumbuffer[work_item_id], framenumber + 1));

    union Color fcolor;
    fcolor.components = (uchar4)(convert_uchar3(res * 255), 1);

    output[work_item_id] = (float3)(x_coord, y_coord, fcolor.c);
}
