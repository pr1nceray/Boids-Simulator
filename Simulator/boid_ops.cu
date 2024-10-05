#include "boid_ops.cuh"

const int num_calc_per_load = 1024;
/*
* Adjust the position based on the rules of the environment.
*/
__global__  void boid_behave(environment local_env, int num_boids, int * boid_locations, half * boid_velocities, float * boid_rotations) 
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //index of thread, as x
	int one_d_idx = gridDim.x * blockDim.x + idx; //1d array index o
  
  int cur_x, cur_y;
  half vel_x, vel_y;

	if (one_d_idx < num_boids)
	{
		cur_x = boid_locations[one_d_idx * 2];
    cur_y = boid_locations[(one_d_idx * 2) + 1];
    half2 tmp = ((half2 * ) boid_velocities)[one_d_idx];
    vel_x = __low2half(tmp);
    vel_y = __high2half(tmp);
	}

  
	//ints cuz x/y of boids are ints
	int center_x_sum = 0, center_y_sum = 0;
	half align_x_sum = 0, align_y_sum = 0;
	size_t align_count = 0;

	int close_x = 0, close_y = 0;
	size_t close_x_count = 0;

  __shared__ int other_locations[num_calc_per_load * 2];
  __shared__ __half other_velocities[num_calc_per_load * 2];

  int num_calc = (((num_boids+( num_calc_per_load - 1))/num_calc_per_load));
	//maybe abstract out to a diff function?
	for (int i = 0; i < num_calc; ++i) {
    __syncthreads();
    if(i * num_calc_per_load + threadIdx.x  < num_boids) 
    {
      other_locations[threadIdx.x] = boid_locations[2 * (i * num_calc_per_load) + threadIdx.x];
      ((half2 *)other_velocities)[threadIdx.x] = ((half2 *) boid_velocities)[i * num_calc_per_load + threadIdx.x];
      other_locations[2 * threadIdx.x] = boid_locations[2 * (i * num_calc_per_load + threadIdx.x)];
    }
    __syncthreads();

    for(int j = 0; j < num_calc_per_load; ++j){
		  if ((i * num_calc_per_load) + j == one_d_idx) 
		  {
			  continue;
		  }

      if( (i * num_calc_per_load) + j > num_boids || one_d_idx >= num_boids)
      {
        break;
      }
      
      int other_x, other_y;
      other_x = other_locations[2 * j];
      other_y = other_locations[2 * j + 1];

      half2 tmp = ((half2 * )other_velocities)[j];
      half other_vx, other_vy;
      other_vx = __low2half(tmp);
      other_vy = __high2half(tmp);

		  float cur_dist = boid_dist(cur_x, cur_y, other_x, other_y);

		  if (cur_dist <= local_env.get_close_range()) {
			  close_x +=  cur_x - other_x;
			  close_y += cur_y - other_y;
			  close_x_count++;
		  }
		  else if (cur_dist <= local_env.get_visible_range()) {
			  align_x_sum += other_vx;
			  align_y_sum += other_vy;
			  align_count++;
			  center_x_sum += other_x;
			  center_y_sum += other_y;
		  }
	  }
  }
	
	if (close_x_count > 0) {
		vel_x += (close_x * local_env.get_avoid_factor()); //normalized.x * cur.speed_max;
		vel_y += close_y * local_env.get_avoid_factor(); //normalized.y * cur.speed_max;
	}


	if (align_count) 
	{
		//alignining with visible
		vel_x += ((align_x_sum / half(align_count)) - vel_x) * half(local_env.get_align_factor());
		vel_y += ((align_y_sum / half(align_count)) - vel_y) * half(local_env.get_align_factor());

		//cohesion with visible
		vel_x += ((static_cast<float>(center_x_sum) / align_count) - cur_x) * local_env.get_center_factor();
		vel_y += ((static_cast<float>(center_y_sum) / align_count) - cur_y) * local_env.get_center_factor();
	}

  if(one_d_idx < num_boids){
    half2 speed = half2(vel_x, vel_y);
	  speed = limit_speed_half(speed);
  
	  speed = adjust_bounds(local_env,cur_x, cur_y, speed);

	  boid_rotations[one_d_idx] = calculate_angle(speed);
    ((half2 *) boid_velocities)[one_d_idx] = speed;
    boid_locations[2 * one_d_idx] = cur_x + __float2int_ru(speed.x);
    boid_locations[2 * one_d_idx + 1] = cur_y + __float2int_ru(speed.y);
  }

}



/*
* Prints the list of boids on the device.
*/

__device__ void print_list_device(environment& env, int num_boids, int * boid_locations, half * boid_velocities, float * boid_rotations)
{

	for (int i = 0; i < num_boids; ++i)
	{
		int cur_x = boid_locations[2 * i];
    int cur_y =  boid_locations[2 * i + 1];
    float rot = boid_rotations[i];
    half2 vel = ((half2 * ) boid_velocities)[i];

		printf("boid %d has x: %d y: %d. vel_x : %.6f, vel_y : %.6f  rot : %.6f\n", 
			i, cur_x, cur_y, __half2float(vel.x), __half2float(vel.y), rot);
	}
}

/*
* Prints the list of boids on the host.
*/
__host__ void print_list_host(environment& env, int num_boids, int * boid_locations, half * boid_velocities, float * boid_rotations)
{
	for (int i = 0; i < num_boids; ++i)
	{
    int cur_x = boid_locations[2 * i];
    int cur_y =  boid_locations[2 * i + 1];
    float rot = boid_rotations[i];
    half2 vel = ((half2 * ) boid_velocities)[i];

		printf("boid %d has x: %d y: %d. vel_x : %.6f, vel_y : %.6f  rot : %.6f\n", 
			i, cur_x, cur_y, __half2float(vel.x), __half2float(vel.y), rot);
	}
}


/*
* Adjust the bounds of the boids to remain within the box.
*/
__device__ __inline__ half2 adjust_bounds(environment& local_env, int cur_x, int cur_y, half2 vel_in)
{

	int margin_x = static_cast<int>(local_env.get_margin_x());
	int margin_y = static_cast<int>(local_env.get_margin_y());
	if (cur_x < margin_x)
	{
		//printf("%.6f\n", cur.vel.x);
		vel_in += half2(local_env.get_turn_factor(), 0);
	}
	else if (cur_x > static_cast<int>(local_env.get_bounds_x() - margin_x))
	{
		//printf("%.6f\n", cur.vel.x);
		vel_in -= half2(local_env.get_turn_factor(), 0);
	}

	if (cur_y < margin_y)
	{
		vel_in += half2(0 , local_env.get_turn_factor() * .5);
	}
	else if (cur_y > static_cast<int>(local_env.get_bounds_y() - margin_y))
	{
		vel_in -= half2(0, local_env.get_turn_factor() * .5);
	}
  return vel_in;
}