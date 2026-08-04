#pragma once

#include <iostream>
#include <vector>
#include <Emitter.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include "../include/FastNoiseLite.h"
#include "MetalContext.hpp"
#include <span>

class MetalGrid
{
    public:
        MetalGrid(std::size_t width, std::size_t height, std::size_t cell_count, std::size_t bytesize, std::vector<Emitter*> emitters, int seed, MetalContext& MetalContext);
        ~MetalGrid();

        //public member variables
        MTL::Buffer* m_pixels;
        float m_buoyancy;
        float m_diff_co;
        float m_viscosity;
        float m_decay;
        float m_curl_mult;
        float m_noise_freq;
        bool m_mult_threaded;

        //public functions
        void update(float dt);
        void reset_density();

        //field getters
        const std::vector<float>& density_r() const;
        const std::vector<float>& density_g() const;
        const std::vector<float>& density_b() const;
        const std::vector<float>& u_velocity() const;
        const std::vector<float>& v_velocity() const;
        const std::vector<float>& get_noise() const;
        std::span<const std::uint8_t> get_pixels() const;

        // vel field setter for test purposes
        void setVelocityAt(std::size_t x, std::size_t y, float u,float v);

        //moved to public for divergence test purposes
        void projectStep();


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

        //kernel encoding functions
        void encodeBuoyancy(MTL::ComputeCommandEncoder* encoder, float dt);
        void encodeDecay(MTL::ComputeCommandEncoder* encoder);
        void encodeEmitter(MTL::ComputeCommandEncoder* encoder, Emitter* emitter);
        void encodePixels(MTL::ComputeCommandEncoder* encoder);
        void encodeDensityAdvection(MTL::ComputeCommandEncoder* encoder, float dt);
        
        

        size_t calcPos(size_t x, size_t y);
        
        void advect_decay_Threaded(float dt, const std::vector<float>& source, std::vector<float>& dest);
        void advect_decay_Rows(float dt, const std::vector<float>& source, std::vector<float>& dest, std::size_t begin, std::size_t end);

        void advectVel_Threaded(float dt);
        void advectVel_Rows(float dt, std::size_t begin, std::size_t end);

        
        
        void swapVel();
        
        void clearPressure();
        
        void calcNoise(float dt);
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
        
        
        void addSource(size_t x, size_t y, float size, sf::Vector3f clr);
        
        void swapDensity();
        
        void diffuse(float dt, const std::vector<float>& source, std::vector<float>& destination);
        void diffuseJacobi_Rows(float a, const std::vector<float>& source, const std::vector<float>& current, std::vector<float>& next, std::size_t begin, std::size_t end);
        void diffuse_Threaded(float a, const std::vector<float>& source, const std::vector<float>& current,  std::vector<float>& next);

        void diffuseVel_Threaded(float dt);
        void diffuseVel_Rows(float dt, std::size_t begin, std::size_t end);

        void densityToPixels();

        void gen_pixels_Threaded();
        void gen_pixels_Rows(std::size_t begin, std::size_t end);

        float density_sample(std::size_t i) const;
        
        void scalarBoundaries(std::vector<float>& field);

        
        //Global variables
        FastNoiseLite m_noise_gen;
        int m_seed;
        std::size_t m_width;
        std::size_t m_height;
        
        std::size_t m_bytesize;
        std::size_t m_cellcount;
        MetalContext& m_metalcontext;
        
        float m_vel_decay;
        float m_source;
        float m_elapsed;
        std::size_t m_pressure_iter;

        //metal kernels
        MTL::ComputePipelineState* m_buoyancyKernel;
        MTL::ComputePipelineState* m_emitterKernel;
        MTL::ComputePipelineState* m_pixelKernel;
        MTL::ComputePipelineState* m_advectKernel;
        
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
        MTL::Buffer* m_density_r;
        MTL::Buffer* m_density_g;
        MTL::Buffer* m_density_b;
        MTL::Buffer* m_u_velocity;
        MTL::Buffer* m_v_velocity;

        std::vector<float> m_diffusion_scratch;
        std::size_t m_diffusion_iterations;

        MTL::Buffer* m_density_r_prev;
        MTL::Buffer* m_density_g_prev;
        MTL::Buffer* m_density_b_prev;
        MTL::Buffer* m_u_velocity_prev;
        MTL::Buffer* m_v_velocity_prev;

        std::vector<float> m_noise;

        std::vector<float> m_pressure;
        std::vector<float> m_pressure_prev;

        std::vector<float> m_divergence;

};
