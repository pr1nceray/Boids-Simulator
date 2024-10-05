#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct vector2f
{
	float x, y;
	vector2f()
		: x(0),y(0)
	{
	}
	vector2f(float x_in, float y_in)
		: x(x_in), y(y_in)
	{
	}
};

__device__ __inline__ vector2f generate_normal(float x, float y) {
	float sqrd = x * x + y * y;
	return { x / sqrd ,y / sqrd };
}


__device__ __inline__ vector2f operator-(const vector2f & lhs, const vector2f & rhs) {
	return { lhs.x - rhs.x, lhs.y - rhs.y };
}
__device__ __inline__ vector2f operator+(const vector2f& lhs, const vector2f& rhs) {
	return { lhs.x + rhs.x, lhs.y + rhs.y };
}