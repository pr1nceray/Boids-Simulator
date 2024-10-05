#include "boid.cuh"
#include "environment.cuh"
#include <iostream>

/*
* Adjust the position based on the rules of the environment.
*/
__global__  void boid_behave(environment local_env, int num_boids, int * boid_locations, half * boid_velocities, float * boid_rotations);

/*
* Prints the list of boids on the device.
*/
__device__ void print_list_device(environment& env,int num_boids, int * boid_locations, half * boid_velocities, float * boid_rotations);

/*
* Prints the list of boids on the host.
*/
__host__ void print_list_host(environment& env, int num_boids, int * boid_locations, half * boid_velocities, float * boid_rotations);

/*
* Ensure that the boid doesn't go out of range
*/
__device__ __inline__ half2 adjust_bounds(environment& local_env, int cur_x, int cur_y, half2 vel_in);