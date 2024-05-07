#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "random"
#include "vector2f.cuh"

/*
* Representation of the boid object
*/
__device__ struct boid
{
	int x, y;

	vector2f vel;
	
	float rot;

	const float speed_max = 9.0;
	const float speed_min = 4.0;
	const float epsilon = 0.000005;

	/*
	* Construct the boid object
	*/
	__host__ __device__ boid()
	{
		x = 600 + (400 - rand() % 800);
		y = 400 + (200 - rand() % 400);
		vel = {1.0f - ((rand() % 500) / 250.0f), 1.0f - ((rand() % 500) / 250.0f) };
		rot = 90;
	}
	
	/*
	* Increase the x,y of the boid based on the velocity.
	*/
	__device__ __inline__ void update_position()
	{
		x += static_cast<int>(vel.x);
		y += static_cast<int>(vel.y);
	}
	
	/*
	* Limit the speed of the boid
	* Boost the speed of the boid if it falls below the minimum.
	*/
	 __device__ __inline__ void limit_speed()
	 {


		float tmp = sqrtf(vel.x * vel.x + vel.y * vel.y);

		if (tmp <= epsilon) { //div by 0 avoidance
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

	 /*
	 * Calculate the angle of the boid given its velocity
	 */
	 __device__ __inline__ void calculate_angle()
	 {	
		 rot = ((atan2f(vel.y,vel.x)) * 180/3.14159265) + 90;
	 }

};

/*
* Euclidian Distance between two boids.
*/
__device__ __inline__ float boid_dist(const boid& param_1,  const boid& param_2)
{
	return sqrtf((param_1.x - param_2.x) * (param_1.x - param_2.x) + (param_1.y - param_2.y) * (param_1.y - param_2.y));

}

