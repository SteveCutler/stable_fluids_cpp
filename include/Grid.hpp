#pragma once

#include <cstddef>
#include <vector>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include "../include/FastNoiseLite.h"
#include "../include/Emitter.hpp"
#include <thread>


class Grid
    {

    public:
        Grid(std::size_t width, std::size_t height, std::vector<Emitter*> emitters, int seed);
        std::vector<std::uint8_t> m_pixels;
        float m_buoyancy;
        float m_diff_co;
        float m_viscosity;
        float m_decay;
        float m_curl_mult;
        float m_noise_freq;
        bool m_mult_threaded;
        
        //multithreading by row template
        template <typename Func>
            void parallelForRows(
                std::size_t y_begin,
                std::size_t y_end,
                Func func){

                    //calc relevant variables for threading
                    std::size_t total_rows = y_end - y_begin;
                
                    //total rows to be processed, leaving out the boundaries
                    std::size_t rows_per_thread = (total_rows + m_thread_count-1)/m_thread_count;
                
                    //creating thread container and reserving space
                    std::vector<std::thread> threads;
                    threads.reserve(m_thread_count);
                
                    for ( std::size_t n = 0; n<m_thread_count; n++){
                        //calc start and end range
                        std::size_t start = y_begin+n*rows_per_thread;
                        std::size_t end_line = start+rows_per_thread;

                        //range check for end line
                        std::size_t end = std::min(end_line,y_end);
                
                        //range check
                        if(start>=end){
                            continue;
                        }
                
                        //start threads and add them to container
                        threads.emplace_back(
                                func,
                                start,
                                end
                            
                        );
                    }

                    //once all threads are created, wait until done
                    for(auto& thread : threads){
                        thread.join();
                    }
                }

        void update(float dt);
        
        void reset_density();

        std::pair<std::size_t,std::size_t> get_xy(std::size_t i);

        //field getters
        const std::vector<float>& density_r() const;
        const std::vector<float>& density_g() const;
        const std::vector<float>& density_b() const;
        const std::vector<float>& u_velocity() const;
        const std::vector<float>& v_velocity() const;
        const std::vector<float>& get_noise() const;

        const std::vector<std::uint8_t> & get_pixels() const;

        //timing getters
        sf::Clock m_performance_clock;
        float time_noise() const;
        float time_vel() const;
        float time_divergence() const;
        float time_pressure() const;
        float time_diffuse() const;
        float time_advect() const;
        float time_advectVel() const;
        float time_project() const;
        float time_addSource() const;


    private:
        size_t calcPos(size_t x, size_t y);
        
        void advect_decay_Threaded(float dt, const std::vector<float>& source, std::vector<float>& dest);
        void advect_decay_Rows(float dt, const std::vector<float>& source, std::vector<float>& dest, std::size_t begin, std::size_t end);

        void advectVel_Threaded(float dt);
        void advectVel_Rows(float dt, std::size_t begin, std::size_t end);

        
        
        void swapVel();
        
        void clearPressure();
        
        void calcNoise_Threaded(float dt);
        void calcNoiseRows(float dt, std::size_t begin, std::size_t end);
        
        void calcVel_Rows(float dt, std::size_t begin, std::size_t end);
        void calcVel_Threaded(float dt);
        
        void calcDivergence_Threaded();
        void calcDivergence_Rows(std::size_t begin, std::size_t end);
        
        //single threaded pressure solve function
        void solvePressure();
        
        //multi threaded implementation
        void solvePressureRows(std::size_t begin, std::size_t end);
        void solvePressureThreaded();
        
        void velBoundaries();
        
        void pressureBoundaries();
        
        void project_Threaded();
        void project_Rows(std::size_t begin, std::size_t end);
        
        void projectStep();
        
        void addSource(size_t x, size_t y, float size, sf::Vector3f clr);
        
        void swapDensity();
        
        void diffuse_Rows(float dt, const std::vector<float>& source, std::vector<float>& dest, std::size_t begin, std::size_t end);
        void diffuse_Threaded(float dt, const std::vector<float>& source, std::vector<float>& dest);

        void diffuseVel_Threaded(float dt);
        void diffuseVel_Rows(float dt, std::size_t begin, std::size_t end);

        void buoyancy(float dt);

        void densityToPixels();

        void gen_pixels_Threaded();
        void gen_pixels_Rows(std::size_t begin, std::size_t end);

        float density_sample(std::size_t i) const;
        

        FastNoiseLite m_noise_gen;




    
    private:

        //Global variables
        int m_seed;
        std::size_t m_width;
        std::size_t m_height;
        
        float m_vel_decay;
        float m_source;
        float m_elapsed;
        std::size_t m_pressure_iter;
        


        //Threading Variables
        std::size_t m_thread_count;
        

        //Timing Debug variables
        float m_noise_ms;
        float m_vel_ms;
        float m_div_ms;
        float m_pressure_ms;
        float m_advect_ms;
        float m_diffuse_ms;
        float m_advectVel_ms;
        float m_project_ms;
        float m_addSource_ms;
        float m_render_ms;

        //Emitters
        std::vector<Emitter*> m_emitters;

        // SoA member variables
        std::vector<float> m_density_r;
        std::vector<float> m_density_g;
        std::vector<float> m_density_b;
        std::vector<float> m_u_velocity;
        std::vector<float> m_v_velocity;

        std::vector<float>m_density_r_prev;
        std::vector<float>m_density_g_prev;
        std::vector<float>m_density_b_prev;
        std::vector<float>m_u_velocity_prev;
        std::vector<float>m_v_velocity_prev;

        std::vector<float> m_noise;

        std::vector<float> m_pressure;
        std::vector<float> m_pressure_prev;

        std::vector<float> m_divergence;

    };
