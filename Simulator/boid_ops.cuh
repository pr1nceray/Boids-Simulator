#include "boid.cuh"
#include "environment.cuh"
#include <iostream>

/*
* Adjust the position based on the rules of the environment.
*/
__global__  void boid_behave(environment local_env, boids_inter boids);

/*
* Prints the list of boids on the device.
*/
__device__ void print_list_device(environment& env, boids_inter &boids);

/*
* Prints the list of boids on the host.
*/
__host__ void print_list_host(environment& env, boids_inter &boids);

/*
* Ensure that the boid doesn't go out of range
*/
__device__ __inline__ void adjust_bounds(environment& local_env, boid& cur);