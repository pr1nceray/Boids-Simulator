#pragma once
#include "boid_ops.cuh"
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Graphics.hpp>


class Window{
	public:
		Window(size_t num_sim_in, int width_in = 1900, int height_in = 999) 
			: num_sim(num_sim_in), width(width_in),height(height_in) 
		{
			window.create(sf::VideoMode(width, height), "Sim Viewer");
			env = environment(num_sim_in, width_in, height_in, 200, 100, 12, 40);
			sprite_ptr = new sf::Sprite[num_sim];
			draw_bounding_box();
			init_sprites(num_sim);
			calc_grids();
			window.setFramerateLimit(60);
		}

		void Display() {
			std::srand(std::time(0));
			boids_inter boids(num_sim); //create and malloc resources.
			while (true) {
				//print_list_host(env, boids);
				get_updates();
				update_boids(boids);
				draw_boids(boids);
			}
		}

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
		int width, height;

		dim3 grid_dim;
		dim3 block_dim;

		/*
		* 
		* 
		* Display the boids
		* Currently trying to offload the work to gpu somehow.
		* 
		*/
		void draw_boids(boids_inter & boids) 
		{
			//
			//
			window.clear();
			window.draw(bounding_box);
			for (size_t i = 0; i < boids.get_boids_len(); ++i) {
				sprite_ptr[i].setPosition(boids.get_boids_host()[i].x, boids.get_boids_host()[i].y);
				sprite_ptr[i].setRotation(boids.get_boids_host()[i].rot );
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
				std::cout << "bad dimeons. replace with error.";
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
		* 
		* 
		* Updates the position of the boids based on the rules and what is currently in the environment 
		*  Currently broken due to certain vars like boids not being accessable (in environment)
		*
		*/
		void update_boids(boids_inter & boids)
		{

			boid_behave <<<grid_dim, block_dim>>> (env, boids);

			cudaError_t err = cudaDeviceSynchronize();
			if (err != 0) {
				std::cout << "an error has occured during sync.";
				exit(1);
			}
			err =  cudaMemcpy(boids.get_boids_host(), boids.get_boids_device(), sizeof(boid) * boids.get_boids_len(), cudaMemcpyDeviceToHost);
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
			if (!txt.loadFromFile("arrow.png"))
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

		void calc_grids()
		{
			block_dim = dim3(8, 8, 1);
			grid_dim = dim3(int(num_sim / 64) + (num_sim %64==0?0:1) ,1,1);

		}

		/*
		* 
		* 
		* 		Update the state of the program based on SFML Events
		* 
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