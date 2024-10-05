#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <iostream>
#include "boid.cuh"


/*
* TODO : fix margin box.
*/

struct environment
{
	private:
	size_t margin_x, margin_y; //number of pixels inbetween the edge of screen and bounding box.
	size_t bounds_x, bounds_y; //x width, y width of simulation

	size_t close_range, visible_range; //range that a boid considers close, and visible

	float avoid_factor, align_factor, center_factor, turn_factor; //factors for calculation


	public:
	
	/*
	* Constructors, Assignment Operator, and Destructors
	*/

	__host__ environment() :
		bounds_x(0), bounds_y(0), close_range(0), visible_range(0), margin_x(0), margin_y(0)
	{
		avoid_factor = .05f;
		align_factor = .05f;
		center_factor = .001f;
		turn_factor = 1;
	}

	__host__ environment(size_t num_boids_in, size_t bounds_x_in = 1200, size_t bounds_y_in = 800,
		size_t margin_x_in = 200, size_t margin_y_in = 100, size_t close_in = 8, size_t visible_in = 60) :
		bounds_x(bounds_x_in), bounds_y(bounds_y_in), margin_x(margin_x_in), margin_y(margin_y_in),
		close_range(close_in), visible_range(visible_in)
	{
		avoid_factor = .05f;
		align_factor = .05f;
		center_factor = .001;
		turn_factor = 1.0f;
	}




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


struct boids_inter {

	public:

  int * boid_locs_h;
  half * boid_velocities_h;
  float * boid_rot_h;


  int * boid_locs_d;
  half *  boid_velocities_d;
  float * boid_rot_d;


	size_t boid_len; //length of boids

  __host__ void alloc_host(size_t num_alloc) {
    const int num_alloc_t2 = num_alloc * 2;
    boid_locs_h = new int[num_alloc_t2];
    boid_velocities_h = new __half[num_alloc_t2];
    boid_rot_h = new float[num_alloc];

    for(size_t i = 0; i < num_alloc; ++i){
      boid_locs_h[i * 2] = 600 + (400 - rand() % 800);
	  	boid_locs_h[(i * 2) + 1] = 400 + (200 - rand() % 400);
		  boid_velocities_h[i * 2] = __float2half(1.0f - ((rand() % 500) / 250.0f));
      boid_velocities_h[(i * 2) + 1] = __float2half(1.0f - ((rand() % 500) / 250.0f));
		  boid_rot_h[i] = 90.0f;
    }
  }
  __host__ void alloc_device(size_t num_alloc){
    const int num_alloc_t2 = num_alloc * 2;

    if(
      cudaMalloc((void **)&boid_locs_d, sizeof(int) * num_alloc_t2) ||
      cudaMalloc((void **)&boid_velocities_d, sizeof(half) * num_alloc_t2) ||
      cudaMalloc((void **)&boid_rot_d, sizeof(float) * num_alloc)
      ){
        std::cout << "error with cuda malloc. CHANGE THIS TO AN ERROR.";
			  exit(1);
    }

    if(
      cudaMemcpy((void*)boid_locs_d, (void*)boid_locs_h, sizeof(int) * num_alloc_t2, cudaMemcpyHostToDevice) ||
      cudaMemcpy((void*)boid_velocities_d, (void*)boid_velocities_h, sizeof(half) * num_alloc_t2, cudaMemcpyHostToDevice) || 
      cudaMemcpy((void*)boid_rot_d, (void*)boid_rot_h, sizeof(float) * num_alloc, cudaMemcpyHostToDevice) 
    ) {
      std::cout << "error with cudaMemcpy. CHANGE THIS TO AN ERROR.";
			exit(1);
    }
    
  }


 	__host__ boids_inter(size_t num_simulate) 
		: boid_len(num_simulate)
	{
		if (boid_len == 0) {
			return;
		}

		alloc_host(num_simulate);
    alloc_device(num_simulate);
	}


	__host__ void delete_dev() {
		if (boid_len == 0) {
			return;
		}
    // new
    cudaFree(boid_locs_d);
    cudaFree(boid_rot_d);
    cudaFree(boid_velocities_d);

    delete[] boid_locs_h;
    delete[] boid_rot_h;
    delete[] boid_velocities_h;


	}

	/*
	* Getters/Setters for the boid variables
	*/

	__inline__ __host__ __device__ size_t get_boids_len()
	{
		return boid_len;
	}
};