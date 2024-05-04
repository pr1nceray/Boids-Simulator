#include "boid.cuh"
#include "environment.cuh"
#include <iostream>

__global__  void boid_behave(environment local_env, boids_inter boids);

__device__ void print_list_device(environment& env, boids_inter &boids);

__host__ void print_list_host(environment& env, boids_inter &boids);

__device__ __inline__ void adjust_bounds(environment& local_env, boid& cur);