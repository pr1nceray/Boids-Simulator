#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "random"
#include "vector2f.cuh"
#include "cuda_fp16.h"

#define speed_max half(9.0)
#define speed_min half(4.0)
#define epsilon half(0.000005)
#define pi 3.14159265
/*
* Representation of the boid object
*/
struct boid
{
	int x, y;

	vector2f vel;
	
	float rot;


	/*
	* Construct the boid object
	*/
	__host__ boid() = delete;
	
	/*
	* Increase the x,y of the boid based on the velocity.
	*/

	/*
	* Limit the speed of the boid
	* Boost the speed of the boid if it falls below the minimum.
	*/

};

/*
* Euclidian Distance between two boids.
*/
__device__ __inline__ float boid_dist(const boid& param_1,  const boid& param_2)
{
	return sqrtf((param_1.x - param_2.x) * (param_1.x - param_2.x) + (param_1.y - param_2.y) * (param_1.y - param_2.y));

}

__device__ __inline__ float boid_dist(const int x1, const int y1, const int x2, const int y2){
  	return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

__device__ __inline__ half2 limit_speed_half(half2 vel)
{
	half tmp = __float2half(sqrtf(vel.x * vel.x + vel.y * vel.y));
	if (tmp <= epsilon) { //div by 0 avoidance
		return vel;
	}

	if (speed_max < tmp)
	{
		vel -= half2((vel.x / tmp) * speed_max, (vel.y / tmp) * speed_max);
	}

	else if (tmp < speed_min)
	{
		vel += half2((vel.x / tmp) * speed_min, (vel.y / tmp) * speed_min);
	}
  return vel;
}

__device__ __inline__ float calculate_angle(half2 vel)
{	
  return ((atan2f(vel.y,vel.x)) * 180/pi) + 90;
}
