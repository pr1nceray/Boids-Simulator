#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "boid.cuh"


/*
* TODO : fix margin box.
*/

__host__ __device__ struct environment
{
	private:
	size_t margin_x, margin_y;
	size_t bounds_x, bounds_y; //x width, y width of simulation

	size_t close_range, visible_range; //range that a boid considers close, and visible

	float avoid_factor, align_factor, center_factor, turn_factor; //factors for calculation


	public:
	
	/*
	* 
	* Constructors, Assignment Operator, and Destructors
	* 
	*/

	__host__ environment() :
		bounds_x(0), bounds_y(0), close_range(0), visible_range(0)
	{
		avoid_factor = .05f;
		align_factor = .05f;
		center_factor = .001f;
		turn_factor = 1;
	}

	__host__ environment(size_t num_boids_in, size_t bounds_x_in = 1200, size_t bounds_y_in = 800,
		size_t margin_x_in = 200, size_t margin_y_in = 100, size_t close_in = 8, size_t visible_in = 40) :
		bounds_x(bounds_x_in), bounds_y(bounds_y_in), margin_x(margin_x_in), margin_y(margin_y_in),
		close_range(close_in), visible_range(visible_in)
	{
		avoid_factor = .05f;
		align_factor = .05f;
		center_factor = .001f;
		turn_factor = .4f;
	}



	/*
	* 
	* Main functions needed
	* 
	*/


	/*
	* 
	* Getters for Environment
	* 
	*/

	__inline__ __host__ __device__ size_t get_margin_x() const
	{
		return margin_x;
	}

	__inline__ __host__ __device__ size_t get_margin_y() const
	{
		return margin_y;
	}

	__inline__ __host__ __device__ size_t get_bounds_x() const
	{
		return bounds_x;
	}
	__inline__ __host__ __device__ size_t get_bounds_y() const
	{
		return bounds_y;
	}

	__inline__ __host__ __device__ size_t get_close_range() const
	{
		return close_range;
	}


	__inline__ __host__ __device__ size_t  get_visible_range() const
	{
		return visible_range;
	}


	__inline__ __host__ __device__ float get_avoid_factor() const
	{
		return avoid_factor;
	}



	 __inline__ __host__ __device__ float get_align_factor() const
	{
		return align_factor;
	}


	__inline__ __host__ __device__ float get_center_factor() const
	{
		return center_factor;
	}

	__inline__ __host__ __device__ float get_turn_factor() const
	{
		return turn_factor;
	}




	/*
	* 
	* Setters for Environmnet
	* 
	*/


	__inline__ __host__ __device__ void set_close_range(size_t range) {
		close_range = range;
	}


	__inline__ __host__ __device__ void set_visible_range(size_t range) {
		visible_range = range;
	}


	__inline__ __host__ __device__ void set_avoid_factor(float factor) {
		avoid_factor = factor;
	}


	__inline__ __host__ __device__ void set_align_factor(float factor) {
		align_factor = factor;
	}


	__inline__ __host__ __device__ void set_center_factor(float factor) {
		center_factor = factor;
	}



};


__host__ __device__ struct boids_inter {

	public:
	
	boid* boids_host; // boids on the host device
	boid* boids_dev; // boids on da gpu.

	size_t boid_len; //length of boids

	
	__host__ boids_inter(size_t num_simulate) 
		: boid_len(num_simulate)
	{
		if (boid_len == 0) {
			return;
		}

		boids_host = new boid[boid_len];

		cudaError_t err;
		err = cudaMalloc((void**)&boids_dev, sizeof(boid) * boid_len);
		if (err != 0) {
			std::cout << "error with cuda malloc. CHANGE THIS TO AN ERROR.";
			exit(1);
		}

		err = cudaMemcpy((void*)boids_dev, (void*)boids_host, sizeof(boid)* boid_len, cudaMemcpyHostToDevice);
		if (err != 0) {
			std::cout << "error with cudaMemcpy. CHANGE THIS TO AN ERROR.";
			exit(1);
		}
	}


	/*
	* Interface functions
	* 
	*/

	__host__ void delete_dev() {
		if (boid_len == 0) {
			return;
		}
		cudaFree(boids_dev);
		delete[] boids_host;
	}

	/*
	* Getters/Setters for the boid variables
	* 
	*/


	__inline__ __host__ __device__ boid* get_boids_device()
	{
		return boids_dev;
	}


	__inline__ __host__ __device__ boid* get_boids_host()
	{
		return boids_host;
	}

	__inline__ __host__ __device__ size_t get_boids_len()
	{
		return boid_len;
	}


};