#pragma once

struct Chunk
{
	union {
		struct { int i, f; };
		int _v[2];
	};

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