#pragma once
#include "boid_ops.cuh"

/*
* Adjust the position based on the rules of the environment.
*/
__global__  void boid_behave(environment local_env, boids_inter boids) 
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //index of thread, as x
	int idy = blockIdx.y * blockDim.y + threadIdx.y; //index of thread, as y

	int one_d_idx = idy * gridDim.x * blockDim.x + idx; //1d array index o

	if (one_d_idx >= boids.get_boids_len())
	{
		return;
	}

	boid & cur = boids.get_boids_device()[one_d_idx];



	//ints cuz x/y of boids are ints
	int center_x_sum = 0, center_y_sum = 0;
	float align_x_sum = 0, align_y_sum = 0;
	size_t align_count = 0;

	int close_x = 0, close_y = 0;
	size_t close_x_count = 0;


	//maybe abstract out to a diff function?
	for (int i = 0; i < boids.get_boids_len(); ++i) {
		if (i == one_d_idx) //i = self
		{
			continue;
		}

		boid & other = boids.get_boids_device()[i];
		float cur_dist = boid_dist(cur,other);

		if (cur_dist <= local_env.get_close_range()) {
			close_x +=  cur.x - other.x;
			close_y += cur.y - other.y;
			close_x_count++;
		}
		else if (cur_dist <= local_env.get_visible_range()) {
			align_x_sum += other.vel.x;
			align_y_sum += other.vel.y;
			align_count++;
			
			center_x_sum += other.x;
			center_y_sum += other.y;
		}
	}
	
	if (close_x_count > 0) {
		cur.vel.x += (close_x * local_env.get_avoid_factor()); //normalized.x * cur.speed_max;
		cur.vel.y += close_y * local_env.get_avoid_factor(); //normalized.y * cur.speed_max;
	}


	if (align_count) 
	{
		//alignining with visible
		cur.vel.x += ((align_x_sum / align_count) - cur.vel.x) * local_env.get_align_factor();
		cur.vel.y += ((align_y_sum / align_count) - cur.vel.y) * local_env.get_align_factor();

		//cohesion with visible
		cur.vel.x += ((static_cast<float>(center_x_sum) / align_count) - cur.x) * local_env.get_center_factor();
		cur.vel.y += ((static_cast<float>(center_y_sum) / align_count) - cur.y) * local_env.get_center_factor();
	}

	cur.limit_speed();
	adjust_bounds(local_env,cur);
	cur.calculate_angle();
	cur.update_position();

}



/*
* Prints the list of boids on the device.
*/

__device__ void print_list_device(environment& env, boids_inter & boids)
{
	for (int i = 0; i < boids.get_boids_len(); ++i)
	{
		boid& cur = boids.get_boids_device()[i];
		printf("boid %d has x: %d y: %d. vel_x : %.6f, vel_y : %.6f  rot : %.6f\n", 
			i, cur.x, cur.y, cur.vel.x, cur.vel.y, cur.rot);
	}
}

/*
* Prints the list of boids on the host.
*/
__device__ void print_list_host(environment& env, boids_inter& boids)
{
	for (int i = 0; i < boids.get_boids_len(); ++i)
	{
		boid& cur = boids.get_boids_host()[i];
		printf("boid %d has x: %d y: %d. vel_x : %.6f, vel_y : %.6f  rot : %.6f\n",
			i, cur.x, cur.y, cur.vel.x, cur.vel.y, cur.rot);
	}
}


/*
* Adjust the bounds of the boids to remain within the box.
*/
__device__ __inline__ void adjust_bounds(environment& local_env, boid & cur)
{

	int margin_x = static_cast<int>(local_env.get_margin_x());
	int margin_y = static_cast<int>(local_env.get_margin_y());
	if (cur.x < margin_x)
	{
		//printf("%.6f\n", cur.vel.x);
		cur.vel.x += local_env.get_turn_factor();
	}
	else if (cur.x > static_cast<int>(local_env.get_bounds_x() - margin_x))
	{
		//printf("%.6f\n", cur.vel.x);
		cur.vel.x -= local_env.get_turn_factor();
	}

	if (cur.y < margin_y)
	{
		cur.vel.y += local_env.get_turn_factor() * .5;
	}
	else if (cur.y > static_cast<int>(local_env.get_bounds_y() - margin_y))
	{
		cur.vel.y -= local_env.get_turn_factor() * .5;
	}

}