#pragma once

struct Chunk
{
	int i;
	int f;
	int dummy0;
	int dummy1;
};

struct Shift
{
	union {
		struct { int dst, src, size; };
		int _v[3];
	};

    Shift(int dst_, int src_, int size_) : dst(dst_), src(src_), size(size_) { }

    Shift() : dst(0), src(0), size(0) { }

};