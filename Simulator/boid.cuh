#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "random"
#include "vector2f.cuh"

__device__ struct boid
{
	int x, y;

	vector2f vel;
	
	float rot;

	const float speed_max = 9.0;
	const float speed_min = 4.0;
	const float epsilon = 0.000005;


	__host__ __device__ boid()
	{
		x = 600 + (400 - rand() % 800);
		y = 400 + (300 - rand() % 600);
		vel = {((rand() % 500) / 250.0f), ((rand() % 500) / 250.0f) };
	}
	
	
	__device__ __inline__ void update_position()
	{
		x += static_cast<int>(vel.x);
		y += static_cast<int>(vel.y);
	}
	
	
	 __device__ __inline__ void limit_speed()
	{


		float tmp = sqrtf(vel.x * vel.x + vel.y * vel.y);

		if (tmp <= epsilon) {
			return;
		}

		if (speed_max < tmp)
		{
			vel.x -= (vel.x / tmp) * speed_max;
			vel.y -= (vel.y / tmp) * speed_max;
		}
		else if (tmp < speed_min)
		{
			vel.x += (vel.x / tmp) * speed_min;
			vel.y += (vel.y / tmp) * speed_min;
		}
	}

	 __device__ __inline__ void calculate_angle()
	 {	
		 /*
		 if (abs(vel.x) <= epsilon) 
		 {
			 rot = vel.y < 0 ? 180:0;
			 return;
		 }
		 if (abs(vel.y) <= epsilon)
		 {
			 rot = vel.x <= 0 ? 270 : 90;
			 return;
		 }
		 */
		 rot = ((atan2f(vel.y,vel.x)) * 180/3.14159265) + 90;
	 }

};

//euclid distance to other boid
__device__ __inline__ float boid_dist(const boid& param_1,  const boid& param_2)
{
	return sqrtf((param_1.x - param_2.x) * (param_1.x - param_2.x) + (param_1.y - param_2.y) * (param_1.y - param_2.y));

}

