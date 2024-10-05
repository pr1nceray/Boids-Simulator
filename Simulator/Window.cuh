#pragma once
#include "boid_ops.cuh"
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Graphics.hpp>


class Window{
	public:

		/*
		* Window Class Constructor 
		* Initializes the window and necessary resources for simulation
		* Doesn't initialize boids. 
		*/
		Window(size_t num_sim_in, int width_in = 1900, int height_in = 1000) 
			: num_sim(num_sim_in), width(width_in),height(height_in) 
		{
			window.create(sf::VideoMode(width, height), "Sim Viewer");
			env = environment(num_sim_in, width_in, height_in, 200, 100, 12, 60);
			sprite_ptr = new sf::Sprite[num_sim];
			draw_bounding_box();
			init_sprites(num_sim);
			calc_grids();
			window.setFramerateLimit(60);
		}

		/*
		* Window Class's Main function.
		* Initialize the boids array
		* Grab the updates for the boids, window, and draw the boids.
		*/
		void Display() {
			std::srand(std::time(0));
			boids_inter boids(num_sim); 
			while (true) {
				get_updates();
				update_boids(boids);
				draw_boids(boids);
			}
		}

		/*
		* Window class destructor
		* 
		*/
		~Window() {
			delete[] sprite_ptr;
		}
	
	private:
		environment env;

		sf::Texture txt;
		sf::RenderWindow window;
		sf::Sprite* sprite_ptr;
		sf::RectangleShape bounding_box;

		size_t num_sim;
		
		size_t width, height;
		size_t box_margin; //uninit;

		dim3 grid_dim;
		dim3 block_dim;

		/*
		* Display the boids
		* Change the sprites position to be x,y that we found.
		* Uses rotation values calculated on gpu.
		*/
		void draw_boids(boids_inter & boids) 
		{
			window.clear();
			window.draw(bounding_box);
			for (size_t i = 0; i < boids.get_boids_len(); ++i) {
				sprite_ptr[i].setPosition(boids.boid_locs_h[2 * i], boids.boid_locs_h[2 * i + 1]);
				sprite_ptr[i].setRotation(boids.boid_rot_h[i]);
				window.draw(sprite_ptr[i]);
			}
			window.display();
		}

		/*
		* Create and display the bounding box that the boids obey
		* 
		*/
		void draw_bounding_box()
		{

			if (width < 200 || height < 200) {
				std::cout << "bad dimensions. replace with error.";
				exit(1);
			}
			bounding_box = sf::RectangleShape(sf::Vector2f(width - 400, height - 200));
			bounding_box.setPosition(200, 100); //half of 400, so we set top left corner to be 200,100.

			bounding_box.setFillColor(sf::Color::Black);
			bounding_box.setOutlineColor(sf::Color::White);
			bounding_box.setOutlineThickness(10);
			
			window.draw(bounding_box);
			window.display();
		}

		/*
		* Updates the position of the boids 
		* TODO : move cuda call to allow for more time between call and sync().
		*/
		void update_boids(boids_inter & boids)
		{

			boid_behave <<<grid_dim, block_dim>>> (env, boids.get_boids_len(), boids.boid_locs_d, boids.boid_velocities_d, boids.boid_rot_d);

			cudaError_t err = cudaDeviceSynchronize();
			if (err != 0) {
				std::cout << "an error has occured during sync.";
				exit(1);
			}
			err =  cudaMemcpy(boids.boid_locs_h, boids.boid_locs_d, sizeof(int) * 2 * boids.get_boids_len(), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cout << "an error has occured during memcpy. Error code : " << err;
				exit(1);
			}
      err =  cudaMemcpy(boids.boid_rot_h, boids.boid_rot_d, sizeof(float) * boids.get_boids_len(), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cout << "an error has occured during memcpy. Error code : " << err;
				exit(1);
			}
		}
		
		/*
		* initialize the sprites to the texture "arrow.png"
		* 
		*/
		void init_sprites(size_t len)
		{
			if (!txt.loadFromFile("./assets/arrow.png"))
			{
				std::cout << "Error with loading image.Replace with error and propper error handling";
				exit(1);
			}
			for (size_t i = 0; i < len; ++i) {
				//sprite_ptr[i].setColor(sf::Color(sf::Color::White));
				sprite_ptr[i].setTexture(txt,true);
				sprite_ptr[i].setOrigin(sprite_ptr[i].getGlobalBounds().width / 2.0, sprite_ptr[i].getGlobalBounds().height / 2.0);
				sprite_ptr[i].setScale(.01, .01);
			}
		}

		/*
		* Calculate the necessary dimensions of the cuda kernel launch
		* 
		*/
		void calc_grids()
		{
			block_dim = dim3(512, 1, 1);
			grid_dim = dim3(int((num_sim +511)/512),1,1);

		}

		/*
		* Update the state of the window based on SFML Events
		*		
		*/
		void get_updates() {
			sf::Event event;
			try {
				window.pollEvent(event);
				if (event.type == sf::Event::Closed) {
					window.close();
					exit(0);
				}
				if (event.type == sf::Event::Resized) {
					//TODO : setters, preventing from being 2 small.
				}
			}
			catch (std::exception e) {
				std::cout << "an exception has occurred while trying to capture SFML events";
			}
	
		}

};