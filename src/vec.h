#pragma once

#include <math.h>

struct Vec3
{
	union {
		struct { float x, y, z, w; };
		float _v[4];
	};

	Vec3(float _x = 0, float _y = 0, float _z = 0, float _w = 0) : x(_x), y(_y), z(_z), w(_w) {}
    Vec3(const Vec3& v, float _w) : x(v.x), y(v.y), z(v.z), w(_w) {}
    Vec3(const Vec3& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
	inline float lengthsq(){ return x*x + y*y + z*z; }
	inline void normalize(){ float norm = sqrtf(x*x + y*y + z*z); x /= norm; y /= norm; z /= norm; }
	inline Vec3& operator+=(const Vec3& v){ x += v.x; y += v.y; z += v.z; return *this; }
	inline Vec3& operator-=(const Vec3& v){ x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline Vec3& operator*=(const float& a){ x *= a; y *= a; z *= a; return *this; }
	inline Vec3& operator*=(const Vec3& v){ x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline Vec3 operator*(float a) const{ return Vec3(x*a, y*a, z*a); }
	inline Vec3 operator/(float a) const{ return Vec3(x / a, y / a, z / a); }
	inline Vec3 operator*(const Vec3& v) const{ return Vec3(x * v.x, y * v.y, z * v.z); }
	inline Vec3 operator+(const Vec3& v) const{ return Vec3(x + v.x, y + v.y, z + v.z); }
	inline Vec3 operator-(const Vec3& v) const{ return Vec3(x - v.x, y - v.y, z - v.z); }
	inline Vec3& operator/=(const float& a){ x /= a; y /= a; z /= a; return *this; }
	inline bool operator!=(const Vec3& v){ return x != v.x || y != v.y || z != v.z; }
    inline float operator[](int i) const{ return i == 0 ? x : (i == 1 ? y : z); }
};




inline Vec3 log(const Vec3& v){ return Vec3(logf(v.x), logf(v.y), logf(v.z)); }
inline Vec3 min3(const Vec3& v1, const Vec3& v2){ return Vec3(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline Vec3 max3(const Vec3& v1, const Vec3& v2){ return Vec3(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline Vec3 cross(const Vec3& v1, const Vec3& v2){ return Vec3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x); }
inline float dot(const Vec3& v1, const Vec3& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline float distancesq(const Vec3& v1, const Vec3& v2){ return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z); }
inline float distance(const Vec3& v1, const Vec3& v2){ return sqrtf((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)); }
