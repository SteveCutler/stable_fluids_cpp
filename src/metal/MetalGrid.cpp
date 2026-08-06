#include "MetalGrid.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include "Emitter.hpp"


MetalGrid::MetalGrid(
    std::size_t width, 
    std::size_t height, 
    std::size_t cell_count, 
    std::size_t bytesize, 
    std::vector<Emitter*> emitters, 
    int seed, 
    MetalContext& MetalContext):


    m_pixels(nullptr),
    m_buoyancy(2.f),
    m_diff_co(4.5f),
    m_viscosity(3.00f),
    m_decay(0.995f),
    m_curl_mult(10),
    m_noise_freq(0.05f),
    m_noiseTimeMult(1.f),
   

    m_noise_strength(2.f),
    m_seed(seed),
    m_width(width),
    m_height(height),
    m_cellcount(cell_count),
    m_bytesize(bytesize),
    m_metalcontext(MetalContext),

    m_vel_decay(0.99f),
    m_source(50.f),
    m_elapsed(0.f),
    m_pressure_iter(20),

    m_thread_count(4),

    m_noise_ms(0.f),
    m_vel_ms(0.f),
    m_div_ms(0.f),
    m_pressure_ms(0.f),
    m_advect_ms(0.f),
    m_diffuse_ms(0.f),
    m_advectVel_ms(0.f),
    m_project_ms(0.f),
    m_addSource_ms(0.f),
    m_render_ms(0.f),
    m_emitters(emitters),

    m_density_r(nullptr),
    m_density_g(nullptr),
    m_density_b(nullptr),
    m_u_velocity(nullptr),
    m_v_velocity(nullptr),
    m_noiseField(nullptr),

    m_diffusion_scratch(width*height,0.f),
    m_diffusion_iterations{20},

    m_advectVelKernel(nullptr),
    m_emitterKernel(nullptr),
    m_pixelKernel(nullptr),
    m_advectKernel(nullptr),
    m_computeNoiseKernel(nullptr),
    m_velFieldKernel(nullptr),
    m_boundaryDensityKernel(nullptr),
    m_boundaryVelocityKernel(nullptr),
    m_boundaryPressureKernel(nullptr),
    m_diffuseVelocityKernel(nullptr),
    m_diffuseDensityKernel(nullptr),

    m_density_r_prev(nullptr),
    m_density_g_prev(nullptr),
    m_density_b_prev(nullptr),
    m_u_velocity_prev(nullptr),
    m_v_velocity_prev(nullptr),

    m_noise(width*height, 0.0f),
    m_pressure(nullptr),
    m_pressure_prev(width*height, 0.0f),
    m_divergence(width*height, 0.0f),
    m_first_frame(true)
    {
        //create error object
         NS::Error* error = nullptr;

        // configure reproducible noise generator
        m_noise_gen.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        m_noise_gen.SetSeed(seed);

        //load compute pipeline kernels
        m_advectVelKernel = m_metalcontext.CreatePipelineState("advectVelocity");
    
        if(m_advectVelKernel == nullptr){
            std::cerr << "problem creating advect velocity kernel" << std::endl;
            return; 
        }

        m_emitterKernel = m_metalcontext.CreatePipelineState("emitter");
    
        if(m_emitterKernel == nullptr){
            std::cerr << "problem creating emitter kernel" << std::endl;
            return; 
        }
        
        m_pixelKernel = m_metalcontext.CreatePipelineState("gen_pixels");
    
        if(m_pixelKernel == nullptr){
            std::cerr << "problem creating gen pixels kernel" << std::endl;
            return; 
        }
        
        m_advectKernel = m_metalcontext.CreatePipelineState("advectDensity");
    
        if(m_advectKernel == nullptr){
            std::cerr << "problem creating advect density kernel" << std::endl;
            return; 
        }

        m_computeNoiseKernel = m_metalcontext.CreatePipelineState("computeNoise");
    
        if(m_computeNoiseKernel == nullptr){
            std::cerr << "problem creating compute noise kernel" << std::endl;
            return; 
        }

        m_velFieldKernel = m_metalcontext.CreatePipelineState("generateVel");
    
        if(m_velFieldKernel == nullptr){
            std::cerr << "problem creating velocityfield kernel" << std::endl;
            return; 
        }

        m_boundaryDensityKernel = m_metalcontext.CreatePipelineState("boundaryDensity");
    
        if(m_boundaryDensityKernel == nullptr){
            std::cerr << "problem creating boundary density kernel" << std::endl;
            return; 
        }

        m_boundaryVelocityKernel = m_metalcontext.CreatePipelineState("boundaryVelocity");
    
        if(m_boundaryVelocityKernel == nullptr){
            std::cerr << "problem creating boundary velocity kernel" << std::endl;
            return; 
        }

        m_boundaryPressureKernel = m_metalcontext.CreatePipelineState("boundaryPressure");
    
        if(m_boundaryPressureKernel == nullptr){
            std::cerr << "problem creating boundary pressure kernel" << std::endl;
            return; 
        }

        m_diffuseVelocityKernel = m_metalcontext.CreatePipelineState("diffuseVelocity");
    
        if(m_diffuseVelocityKernel == nullptr){
            std::cerr << "problem creating velocity diffusion kernel" << std::endl;
            return; 
        }

        // m_diffuseDensityKernel = m_metalcontext.CreatePipelineState("diffuseDensity");
    
        // if(m_boundaryPressureKernel == nullptr){
        //     std::cerr << "problem creating density diffusion kernel" << std::endl;
        //     return; 
        // }

        // initialize buffers
        m_density_r = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_density_g = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_density_b = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_v_velocity = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_u_velocity = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_noiseField = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_pressure = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        
        //prev frame buffers
        m_density_r_prev = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_density_g_prev = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_density_b_prev = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_v_velocity_prev = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_u_velocity_prev = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );

        //setting pixel buffer to correct size for all 4 channels

        std::size_t pixelBytes = m_cellcount * 4 * sizeof(std::uint8_t);

        m_pixels = m_metalcontext.get_device()->newBuffer(
            pixelBytes,
            MTL::ResourceStorageModeShared
        );

        
        // if any error in creating the buffers: printout error, call destructor, shut it down
        if (m_density_r == nullptr ||
            m_density_g == nullptr || 
            m_density_b == nullptr ||
            m_v_velocity == nullptr ||
            m_u_velocity == nullptr ||
            m_noiseField == nullptr ||
            m_pressure == nullptr ||
            m_density_r_prev == nullptr ||
            m_density_g_prev == nullptr || 
            m_density_b_prev == nullptr ||
            m_v_velocity_prev == nullptr ||
            m_u_velocity_prev == nullptr ||
            m_pixels == nullptr) {
                
                std::cerr << "Error allocating Metal buffers.\n";
                
                //if buffer didn't err out release memory before end
                if (m_density_r != nullptr) m_density_r->release();
                if (m_density_g != nullptr) m_density_g->release();
                if (m_density_b != nullptr) m_density_b->release();
                if (m_u_velocity != nullptr) m_u_velocity->release();
                if (m_v_velocity != nullptr) m_v_velocity->release();
                if (m_noiseField != nullptr) m_noiseField->release();
                if (m_pressure != nullptr) m_noiseField->release();
                if (m_density_r_prev != nullptr) m_density_r->release();
                if (m_density_g_prev != nullptr) m_density_g->release();
                if (m_density_b_prev != nullptr) m_density_b->release();
                if (m_u_velocity_prev != nullptr) m_u_velocity->release();
                if (m_v_velocity_prev != nullptr) m_v_velocity->release();
                if (m_pixels != nullptr) m_pixels->release();
                
                return;
            }
            
        //initialize buffer values
        float* r = static_cast<float*>(m_density_r->contents());
        float* g = static_cast<float*>(m_density_g->contents());
        float* b = static_cast<float*>(m_density_b->contents());
        float* u = static_cast<float*>(m_u_velocity->contents());
        float* v = static_cast<float*>(m_v_velocity->contents());
        float* nf = static_cast<float*>(m_noiseField->contents());
        float* pr = static_cast<float*>(m_pressure->contents());
        float* r_prev = static_cast<float*>(m_density_r_prev->contents());
        float* g_prev = static_cast<float*>(m_density_g_prev->contents());
        float* b_prev = static_cast<float*>(m_density_b_prev->contents());
        float* u_prev = static_cast<float*>(m_u_velocity_prev->contents());
        float* v_prev = static_cast<float*>(m_v_velocity_prev->contents());
        std::uint8_t* p = static_cast<std::uint8_t*>(m_pixels->contents());

        std::fill_n(r,m_cellcount,0.f);
        std::fill_n(g,m_cellcount,0.f);
        std::fill_n(b,m_cellcount,0.f);
        //preset velocity values for testing
        std::fill_n(u,m_cellcount,0.f);
        std::fill_n(v,m_cellcount,0.f);
        std::fill_n(nf,m_cellcount,0.f);
        std::fill_n(pr,m_cellcount,0.f);
        std::fill_n(r_prev,m_cellcount,0.f);
        std::fill_n(g_prev,m_cellcount,0.f);
        std::fill_n(b_prev,m_cellcount,0.f);
        std::fill_n(u_prev,m_cellcount,0.f);
        std::fill_n(v_prev,m_cellcount,0.f);
        std::fill_n(p,m_cellcount*4,0u);
            
       

    };

void MetalGrid::update(float dt){

    m_elapsed += dt;

    MTL::CommandBuffer* commandBuffer = m_metalcontext.get_commandqueue()->commandBuffer();

    if(commandBuffer == nullptr){
        std::cerr << "Error creating command buffer" << std::endl;
        return;
    }

    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    if (encoder == nullptr) {
        std::cerr << "Error creating compute encoder.\n";
        return;
    }

    //kernel order for CPU match
    /*
    calc noise
    calc vel
    swap vel
    advect vel
    vel boundaries
    swap vel
    diffuse vel
    vel boundaries
    project step
    add source
    swap density
    diffuse density
    swap density
    advect density
    density boundaries
    */

    //create noise field
    encodeNoiseField(encoder, m_elapsed);

    //create curl vel field
    encodeVelField(encoder, dt);

    //swap for advection input
    std::swap(m_u_velocity,m_u_velocity_prev);
    std::swap(m_v_velocity,m_v_velocity_prev);
    
    //encode advect vel
    encodeAdvectVel(encoder, dt);

    //encode velocity boundaries
    encodeBoundaryVelocity(encoder);
    
    std::swap(m_u_velocity,m_u_velocity_prev);
    std::swap(m_v_velocity,m_v_velocity_prev);
    
    //diffuse velocity
    encodeDiffuseVelocity(encoder, dt);
    
    //encode boundary velocity
    //encodeBoundaryVelocity(encoder);

    
    //encode emitter kernel
    for(auto& emitter : m_emitters){
        encodeEmitter(encoder, emitter);
    }
    //swapping buffers to prepare for density advection
    std::swap(m_density_r, m_density_r_prev);
    std::swap(m_density_g, m_density_g_prev);
    std::swap(m_density_b, m_density_b_prev);
    
    //encode density advection
    encodeDensityAdvection(encoder, dt);

    //density boundary
    encodeBoundaryDensity(encoder);
   
    //encode pixels
    encodePixels(encoder);


    //end encoding for kernel
    encoder->endEncoding();

    //submit the work and wait for completion.

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        auto* error = commandBuffer->error();

        std::cerr << "Metal command buffer failed";

        if (error != nullptr) {
            std::cerr << ": "
                    << error->localizedDescription()->utf8String();
        }

        std::cerr << '\n';
    }

};

void MetalGrid::encodeNoiseField(MTL::ComputeCommandEncoder* encoder, float elapsed){

    encoder->setComputePipelineState(m_computeNoiseKernel);

    //convert width and height to uint for passing to kernel
    const std::uint32_t width = static_cast<std::uint32_t>(m_width);
    const std::uint32_t height = static_cast<std::uint32_t>(m_height);

    //bind the buffers
    encoder->setBuffer(m_noiseField, 0, 0);

    encoder->setBytes(
        &elapsed,
        sizeof(float),
        1
    );
    encoder->setBytes(
        &m_noise_freq,
        sizeof(float),
        2
    );
    encoder->setBytes(
        &m_noiseTimeMult,
        sizeof(float),
        3
    );

     encoder->setBytes(
        &width,
        sizeof(width),
        4
    );

    encoder->setBytes(
        &height,
        sizeof(height),
        5
    );
    
   
    std::uint32_t gpuCellCount = static_cast<std::uint32_t>(m_cellcount);

    encoder->setBytes(
        &gpuCellCount,
        sizeof(gpuCellCount),
        6
    );




    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);

    MTL::Size threadgroupSize(16, 16, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeVelField(MTL::ComputeCommandEncoder* encoder, float dt){

    encoder->setComputePipelineState(m_velFieldKernel);

    //convert width and height to uint for passing to kernel
    const std::uint32_t width = static_cast<std::uint32_t>(m_width);
    const std::uint32_t height = static_cast<std::uint32_t>(m_height);

    //bind the buffers
    encoder->setBuffer(m_noiseField, 0, 0);
    encoder->setBuffer(m_u_velocity, 0, 1);
    encoder->setBuffer(m_v_velocity, 0, 2);
    encoder->setBuffer(m_density_r, 0, 3);
    encoder->setBuffer(m_density_g, 0, 4);
    encoder->setBuffer(m_density_b, 0, 5);

    encoder->setBytes(
        &dt,
        sizeof(float),
        6
    );
    encoder->setBytes(
        &m_curl_mult,
        sizeof(float),
        7
    );
    encoder->setBytes(
        &width,
        sizeof(width),
        8
    );

     encoder->setBytes(
        &height,
        sizeof(height),
        9
    );


   
    std::uint32_t gpuCellCount = static_cast<std::uint32_t>(m_cellcount);

    encoder->setBytes(
        &gpuCellCount,
        sizeof(gpuCellCount),
        10
    );

    encoder->setBytes(
        &m_noise_strength,
        sizeof(float),
        11
    );




    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);

    MTL::Size threadgroupSize(16, 16, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeAdvectVel(MTL::ComputeCommandEncoder* encoder, float dt){
    //set the buoyancy kernel to the encoder
    encoder->setComputePipelineState(m_advectVelKernel);

    //convert width and height to uint for passing to kernel
    const std::uint32_t width = static_cast<std::uint32_t>(m_width);
    const std::uint32_t height = static_cast<std::uint32_t>(m_height);

    //bind the buffers
    encoder->setBuffer(m_u_velocity, 0, 0);
    encoder->setBuffer(m_v_velocity, 0, 1);
    encoder->setBuffer(m_u_velocity_prev, 0, 2);
    encoder->setBuffer(m_v_velocity_prev, 0, 3);


    encoder->setBytes(
        &dt,
        sizeof(float),
        4
    );
    encoder->setBytes(
        &m_buoyancy,
        sizeof(float),
        5
    );

    //convert std::size_t to uint for metal

    std::uint32_t gpuCellCount = static_cast<std::uint32_t>(m_cellcount);

    encoder->setBytes(
        &gpuCellCount,
        sizeof(gpuCellCount),
        6
    );


    encoder->setBytes(
        &width,
        sizeof(width),
        7
    );

    encoder->setBytes(
        &height,
        sizeof(height),
        8
    );

    encoder->setBytes(
        &m_vel_decay,
        sizeof(m_vel_decay),
        9
    );
    encoder->setBuffer(m_density_r, 0, 10);
    encoder->setBuffer(m_density_g, 0, 11);
    encoder->setBuffer(m_density_b, 0, 12);


    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);


    MTL::Size threadgroupSize(16, 16, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeEmitter(MTL::ComputeCommandEncoder* encoder, Emitter* emitter){
    
    encoder->setComputePipelineState(m_emitterKernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);
    
    
    const std::uint32_t local_x = static_cast<std::uint32_t>(std::round(emitter->m_pos.x));
    const std::uint32_t local_y = static_cast<std::uint32_t>(std::round(emitter->m_pos.y));
    const std::uint32_t diameter = static_cast<std::uint32_t>((std::ceil(emitter->m_rad) * 2) +1);
    
    const std::uint32_t width = static_cast<std::uint32_t>(m_width);
    const std::uint32_t height = static_cast<std::uint32_t>(m_height);


    encoder->setBytes(
        &local_x,
        sizeof(local_x),
        3
    );

    encoder->setBytes(
        &local_y,
        sizeof(local_y),
        4
    );

    encoder->setBytes(
        &emitter->m_rad,
        sizeof(float),
        5
    );

    encoder->setBytes(
        &emitter->m_clr.x,
        sizeof(float),
        6
    );
    encoder->setBytes(
        &emitter->m_clr.y,
        sizeof(float),
        7
    );
    encoder->setBytes(
        &emitter->m_clr.z,
        sizeof(float),
        8
    );

    encoder->setBytes(
        &diameter,
        sizeof(diameter),
        9
    );

    encoder->setBytes(
        &width,
        sizeof(width),
        10
    );
    encoder->setBytes(
        &height,
        sizeof(height),
        11
    );

    //set grid size for kernel
    MTL::Size gridSize(diameter, diameter, 1);

    MTL::Size threadgroupSize(8, 8, 1);

    //dispatch threads
    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );

};

void MetalGrid::encodeDensityAdvection(MTL::ComputeCommandEncoder* encoder, float dt){
     encoder->setComputePipelineState(m_advectKernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);
    encoder->setBuffer(m_density_r_prev, 0, 3);
    encoder->setBuffer(m_density_g_prev, 0, 4);
    encoder->setBuffer(m_density_b_prev, 0, 5);
    encoder->setBuffer(m_u_velocity, 0, 6);
    encoder->setBuffer(m_v_velocity, 0, 7);
    
    
    
    const std::uint32_t width = static_cast<std::uint32_t>(m_width);
    const std::uint32_t height = static_cast<std::uint32_t>(m_height);


    encoder->setBytes(
        &dt,
        sizeof(dt),
        8
    );
    encoder->setBytes(
        &m_cellcount,
        sizeof(m_cellcount),
        9
    );


    encoder->setBytes(
        &width,
        sizeof(width),
        10
    );
    encoder->setBytes(
        &height,
        sizeof(height),
        11
    );

    encoder->setBytes(
        &m_decay,
        sizeof(m_decay),
        12
    );

    //set grid size for kernel
    MTL::Size gridSize(width, height, 1);

    MTL::Size threadgroupSize(16, 16, 1);

    //dispatch threads
    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
}

void MetalGrid::encodePixels(MTL::ComputeCommandEncoder* encoder){
   
    encoder->setComputePipelineState(m_pixelKernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);
    encoder->setBuffer(m_pixels, 0, 3);


    //convert std::size_t to uint for metal

    std::uint32_t gpuCellCount = static_cast<std::uint32_t>(m_cellcount);

    encoder->setBytes(
        &gpuCellCount,
        sizeof(gpuCellCount),
        4
    );

    //set number of threads for kernel
    MTL::Size gridSize(m_cellcount, 1, 1);

    const std::size_t threadgroupWidth =
        std::min(
            m_pixelKernel->maxTotalThreadsPerThreadgroup(),
            m_cellcount
        );

    MTL::Size threadgroupSize(threadgroupWidth, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeBoundaryDensity(MTL::ComputeCommandEncoder* encoder){
   
    encoder->setComputePipelineState(m_boundaryDensityKernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);

    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);
    

    encoder->setBytes(&width,
        sizeof(width),
        3);
    encoder->setBytes(&height,sizeof(height),4);
    encoder->setBytes(&m_cellcount,sizeof(m_cellcount),5);

    std::uint32_t edge_length =  static_cast<std::uint32_t>(std::max(width,height));


    //set number of threads for kernel
    MTL::Size gridSize(edge_length, 1, 1);


    MTL::Size threadgroupSize(256, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeBoundaryVelocity(MTL::ComputeCommandEncoder* encoder){
   
    encoder->setComputePipelineState(m_boundaryVelocityKernel);

    //bind the buffers
    encoder->setBuffer(m_u_velocity, 0, 0);
    encoder->setBuffer(m_v_velocity, 0, 1);

    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);
    

    encoder->setBytes(&width,
        sizeof(width),
        2);

    encoder->setBytes(&height,sizeof(height),3);
    encoder->setBytes(&m_cellcount,sizeof(m_cellcount),4);

    std::uint32_t edge_length =  static_cast<std::uint32_t>(std::max(width,height));


    //set number of threads for kernel
    MTL::Size gridSize(edge_length, 1, 1);


    MTL::Size threadgroupSize(256, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeDiffuseVelocity(MTL::ComputeCommandEncoder* encoder, float dt){
   
    encoder->setComputePipelineState(m_diffuseVelocityKernel);

    //bind the buffers
    encoder->setBuffer(m_u_velocity, 0, 0);
    encoder->setBuffer(m_v_velocity, 0, 1);
    encoder->setBuffer(m_u_velocity_prev, 0, 2);
    encoder->setBuffer(m_v_velocity_prev, 0, 3);

    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);
    

    encoder->setBytes(&dt,
        sizeof(dt),
        4);
    encoder->setBytes(&m_viscosity,
        sizeof(m_viscosity),
        5);

    encoder->setBytes(&width,
        sizeof(width),
        6);
    encoder->setBytes(&height,
        sizeof(height),
        7);
    encoder->setBytes(&total,
        sizeof(total),
        8);

    std::uint32_t edge_length =  static_cast<std::uint32_t>(std::max(width,height));


    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);


    MTL::Size threadgroupSize(16, 16, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

std::span<const std::uint8_t> MetalGrid::get_pixels() const
{
    auto* pixels =  static_cast<const std::uint8_t*>(
        m_pixels->contents()
    );

    return {pixels, m_width * m_height *4};
}


// void MetalGrid::reset_density();

// //field getters
// const std::vector<float>& MetalGrid::density_r(){
    
// } const;
// const std::vector<float>& MetalGrid::density_g() const;
// const std::vector<float>& MetalGrid::density_b() const;
// const std::vector<float>& MetalGrid::u_velocity() const;
// const std::vector<float>& MetalGrid::v_velocity() const;
// const std::vector<float>& MetalGrid::get_noise() const;
// const std::vector<std::uint8_t> & MetalGrid::get_pixels() const;

// // vel field setter for test purposes
// void MetalGrid::setVelocityAt(std::size_t x, std::size_t y, float u,float v);

// //moved to public for divergence test purposes
// void MetalGrid::projectStep();


// //timing getters
// sf::Clock m_performance_clock;
// float MetalGrid::time_noise() const;
// float MetalGrid::time_vel() const;
// float MetalGrid::time_divergence() const;
// float MetalGrid::time_pressure() const;
// float MetalGrid::time_diffuse() const;
// float MetalGrid::time_advect() const;
// float MetalGrid::time_advectVel() const;
// float MetalGrid::time_project() const;
// float MetalGrid::time_addSource() const;

// void MetalGrid::advect_decay_Threaded(float dt, const std::vector<float>& source, std::vector<float>& dest);
// void MetalGrid::advect_decay_Rows(float dt, const std::vector<float>& source, std::vector<float>& dest, std::size_t begin, std::size_t end);

// void MetalGrid::advectVel_Threaded(float dt);
// void MetalGrid::advectVel_Rows(float dt, std::size_t begin, std::size_t end);



// void MetalGrid::swapVel();

// void MetalGrid::clearPressure();

// void MetalGrid::calcNoise(float dt);
// void MetalGrid::calcNoise_Threaded(float dt);
// void MetalGrid::calcNoiseRows(float dt, std::size_t begin, std::size_t end);

// void MetalGrid::calcVel_Rows(float dt, std::size_t begin, std::size_t end);
// void MetalGrid::calcVel_Threaded(float dt);

// void MetalGrid::calcDivergence_Threaded();
// void MetalGrid::calcDivergence_Rows(std::size_t begin, std::size_t end);

// //single threaded pressure solve function
// void MetalGrid::solvePressure();

// //multi threaded implementation
// void MetalGrid::solvePressureRows(std::size_t begin, std::size_t end);
// void MetalGrid::solvePressureThreaded();

// void MetalGrid::velBoundaries();

// void MetalGrid::pressureBoundaries();

// void MetalGrid::project_Threaded();
// void MetalGrid::project_Rows(std::size_t begin, std::size_t end);


// void MetalGrid::addSource(size_t x, size_t y, float size, sf::Vector3f clr);

// void MetalGrid::swapDensity();

// void MetalGrid::diffuse(float dt, const std::vector<float>& source, std::vector<float>& destination);
// void MetalGrid::diffuseJacobi_Rows(float a, const std::vector<float>& source, const std::vector<float>& current, std::vector<float>& next, std::size_t begin, std::size_t end);
// void MetalGrid::diffuse_Threaded(float a, const std::vector<float>& source, const std::vector<float>& current,  std::vector<float>& next);

// void MetalGrid::diffuseVel_Threaded(float dt);
// void MetalGrid::diffuseVel_Rows(float dt, std::size_t begin, std::size_t end);

// void MetalGrid::densityToPixels();

// void MetalGrid::gen_pixels_Threaded();
// void MetalGrid::gen_pixels_Rows(std::size_t begin, std::size_t end){
//     for (std::size_t y = begin; y<end; y++){
//             std::size_t row = y*m_width;

//             for( std::size_t x=0; x<m_width; x++){
//                 std::size_t i = row+x;
                
//                 //clamp between 0 and 1
//                 float d_r = m_density_r[i];
//                 float d_g = m_density_g[i];
//                 float d_b = m_density_b[i];
                
                
//                 //convert density to RGB values
//                 std::uint8_t value_r = static_cast<std::uint8_t>(d_r * 255.f);
//                 std::uint8_t value_g = static_cast<std::uint8_t>(d_g * 255.f);
//                 std::uint8_t value_b = static_cast<std::uint8_t>(d_b * 255.f);
                
//                 //find correct position in pixel array, given each pixel has 1 components
//                 std::size_t p = i * 4;
                
//                 //create greyscale image with alpha of 1
//                 m_pixels[p] = value_r;  //R
//                 m_pixels[p+1] = value_g;//G
//                 m_pixels[p+2] = value_b;//B
//                 m_pixels[p+3] = 255;  //Alpha channel
//             }            
//     }    
// };

// float MetalGrid::density_sample(std::size_t i) const;

// void MetalGrid::scalarBoundaries(std::vector<float>& field);

MetalGrid::~MetalGrid()
{
    if (m_density_r != nullptr) {
        m_density_r->release();
        m_density_r = nullptr;
    }

    if (m_density_g != nullptr) {
        m_density_g->release();
        m_density_g = nullptr;
    }

    if (m_density_b != nullptr) {
        m_density_b->release();
        m_density_b = nullptr;
    }

    if (m_u_velocity != nullptr) {
        m_u_velocity->release();
        m_u_velocity = nullptr;
    }

    if (m_v_velocity != nullptr) {
        m_v_velocity->release();
        m_v_velocity = nullptr;
    }


    std::cout << "Metal grid destroyed and resources released" << std::endl;

}
