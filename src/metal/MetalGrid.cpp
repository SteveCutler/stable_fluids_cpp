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
    m_buoyancy(20.f),
    m_diff_co(1.5f),
    m_viscosity(0.00f),
    m_decay(0.995f),
    m_curl_mult(10),
    m_noise_freq(0.05f),
    m_noiseTimeMult(1.f),
    m_noise_strength(2.f),

    m_gpuexecutiontime(0),
    m_cpuwaittime(0),
   
    m_u_velocity(nullptr),
    m_v_velocity(nullptr),

    m_seed(seed),
    m_width(width),
    m_height(height),
    m_cellcount(cell_count),
    m_bytesize(bytesize),
    m_metalcontext(MetalContext),

    m_vel_decay(0.994f),
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
    m_noiseField(nullptr),
    m_diffusion_scratch_r(nullptr),
    m_diffusion_scratch_g(nullptr),
    m_diffusion_scratch_b(nullptr),

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
    m_divergenceKernel(nullptr),
    m_solvePressureKernel(nullptr),
    m_projectKernel(nullptr),
    m_copyDensityKernel(nullptr),

    m_density_r_prev(nullptr),
    m_density_g_prev(nullptr),
    m_density_b_prev(nullptr),
    m_u_velocity_prev(nullptr),
    m_v_velocity_prev(nullptr),
    
    m_divergence(nullptr),
    m_pressure(nullptr),
    m_pressure_prev(nullptr),

    m_noise(width*height, 0.0f),


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

        m_diffuseDensityKernel = m_metalcontext.CreatePipelineState("diffuseDensity");
    
        if(m_diffuseDensityKernel == nullptr){
            std::cerr << "problem creating density diffusion kernel" << std::endl;
            return; 
        }

        
        m_copyDensityKernel = m_metalcontext.CreatePipelineState("copyDensity");
    
        if(m_copyDensityKernel == nullptr){
            std::cerr << "problem creating copy density kernel" << std::endl;
            return; 
        }

        m_divergenceKernel = m_metalcontext.CreatePipelineState("computeDivergence");
    
        if(m_divergenceKernel == nullptr){
            std::cerr << "problem creating divergence kernel" << std::endl;
            return; 
        }

        
        m_solvePressureKernel = m_metalcontext.CreatePipelineState("solvePressure");
    
        if(m_solvePressureKernel == nullptr){
            std::cerr << "problem creating pressure solver kernel" << std::endl;
            return; 
        }

        m_projectKernel = m_metalcontext.CreatePipelineState("project");
    
        if(m_projectKernel == nullptr){
            std::cerr << "problem creating projection kernel" << std::endl;
            return; 
        }

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
        m_diffusion_scratch_r = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_diffusion_scratch_g = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );
        m_diffusion_scratch_b = m_metalcontext.get_device()->newBuffer(
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

        m_divergence = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );

        m_pressure = m_metalcontext.get_device()->newBuffer(
            m_bytesize,
            MTL::ResourceStorageModeShared
        );

        m_pressure_prev = m_metalcontext.get_device()->newBuffer(
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
            m_diffusion_scratch_r == nullptr ||
            m_diffusion_scratch_g == nullptr ||
            m_diffusion_scratch_b == nullptr ||
            m_pressure == nullptr ||
            m_density_r_prev == nullptr ||
            m_density_g_prev == nullptr || 
            m_density_b_prev == nullptr ||
            m_v_velocity_prev == nullptr ||
            m_u_velocity_prev == nullptr ||
            m_divergence == nullptr ||
            m_pressure == nullptr ||
            m_pressure_prev == nullptr ||
            m_pixels == nullptr) {
                
                std::cerr << "Error allocating Metal buffers.\n";
                
                //if buffer didn't err out release memory before end
                if (m_density_r != nullptr) m_density_r->release();
                if (m_density_g != nullptr) m_density_g->release();
                if (m_density_b != nullptr) m_density_b->release();
                if (m_u_velocity != nullptr) m_u_velocity->release();
                if (m_v_velocity != nullptr) m_v_velocity->release();
                if (m_noiseField != nullptr) m_noiseField->release();
                if (m_diffusion_scratch_r != nullptr) m_noiseField->release();
                if (m_diffusion_scratch_g != nullptr) m_noiseField->release();
                if (m_diffusion_scratch_b != nullptr) m_noiseField->release();
                if (m_pressure != nullptr) m_noiseField->release();
                if (m_density_r_prev != nullptr) m_density_r->release();
                if (m_density_g_prev != nullptr) m_density_g->release();
                if (m_density_b_prev != nullptr) m_density_b->release();
                if (m_u_velocity_prev != nullptr) m_u_velocity->release();
                if (m_v_velocity_prev != nullptr) m_v_velocity->release();
                if (m_divergence != nullptr) m_divergence->release();
                if (m_pressure != nullptr) m_pressure->release();
                if (m_pressure_prev != nullptr) m_pressure_prev->release();
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
        float* ds_r = static_cast<float*>(m_diffusion_scratch_r->contents());
        float* ds_g = static_cast<float*>(m_diffusion_scratch_g->contents());
        float* ds_b = static_cast<float*>(m_diffusion_scratch_b->contents());
        float* pr = static_cast<float*>(m_pressure->contents());
        float* r_prev = static_cast<float*>(m_density_r_prev->contents());
        float* g_prev = static_cast<float*>(m_density_g_prev->contents());
        float* b_prev = static_cast<float*>(m_density_b_prev->contents());
        float* u_prev = static_cast<float*>(m_u_velocity_prev->contents());
        float* v_prev = static_cast<float*>(m_v_velocity_prev->contents());
        float* div = static_cast<float*>(m_divergence->contents());
        float* pres = static_cast<float*>(m_pressure->contents());
        float* pres_prev = static_cast<float*>(m_pressure_prev->contents());
        std::uint8_t* p = static_cast<std::uint8_t*>(m_pixels->contents());

        std::fill_n(r,m_cellcount,0.f);
        std::fill_n(g,m_cellcount,0.f);
        std::fill_n(b,m_cellcount,0.f);
        //preset velocity values for testing
        std::fill_n(u,m_cellcount,0.f);
        std::fill_n(v,m_cellcount,0.f);
        std::fill_n(nf,m_cellcount,0.f);
        std::fill_n(ds_r,m_cellcount,0.f);
        std::fill_n(ds_g,m_cellcount,0.f);
        std::fill_n(ds_b,m_cellcount,0.f);
        std::fill_n(pr,m_cellcount,0.f);
        std::fill_n(r_prev,m_cellcount,0.f);
        std::fill_n(g_prev,m_cellcount,0.f);
        std::fill_n(b_prev,m_cellcount,0.f);
        std::fill_n(u_prev,m_cellcount,0.f);
        std::fill_n(v_prev,m_cellcount,0.f);
        std::fill_n(div,m_cellcount,0.f);
        std::fill_n(pres,m_cellcount,0.f);
        std::fill_n(pres_prev,m_cellcount,0.f);
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
    encodeBoundaryVelocity(encoder);

    //compute divergence
    encodeComputeDivergence(encoder);

    //compute Pressure
    solvePressureHelper(encoder);

    encodeBoundaryPressure(encoder);

    encodeProjectPressure(encoder);

    encodeBoundaryVelocity(encoder);


    
    //encode emitter kernel
    for(auto& emitter : m_emitters){
        encodeEmitter(encoder, emitter);
    }
    //swapping buffers to prepare for density advection
    std::swap(m_density_r, m_density_r_prev);
    std::swap(m_density_g, m_density_g_prev);
    std::swap(m_density_b, m_density_b_prev);

    densityDiffusionHelper(encoder, dt);

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
    const auto waitStart = std::chrono::steady_clock::now();
    
    commandBuffer->waitUntilCompleted();
    const auto waitEnd = std::chrono::steady_clock::now();

    //gpu computation time logic
    m_gpuexecutiontime = (commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime()) * 1000;
    
    //gpu computation time logic
    m_cpuwaittime = std::chrono::duration<double, std::milli>((waitEnd - waitStart)).count();
    
    
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

//clear buffers
void MetalGrid::ClearBuffers(){
    MTL::CommandBuffer* commandBuffer = m_metalcontext.get_commandqueue()->commandBuffer();

    if(commandBuffer == nullptr){
        std::cerr << "Error creating command buffer" << std::endl;
        return;
    }

    MTL::BlitCommandEncoder* blit = commandBuffer->blitCommandEncoder();

    if (blit == nullptr) {
        std::cerr << "Error creating compute encoder.\n";
        return;
    }

    //density
    blit->fillBuffer(
        m_density_r,
        NS::Range::Make(0, m_density_r->length()),
        0);

    blit->fillBuffer(
        m_density_g,
        NS::Range::Make(0, m_density_g->length()),
        0);

    blit->fillBuffer(
        m_density_b,
        NS::Range::Make(0, m_density_b->length()),
        0);
    
    //density prev
    blit->fillBuffer(
        m_density_r_prev,
        NS::Range::Make(0, m_density_r_prev->length()),
        0);

    blit->fillBuffer(
        m_density_g_prev,
        NS::Range::Make(0, m_density_g_prev->length()),
        0);

    blit->fillBuffer(
        m_density_b_prev,
        NS::Range::Make(0, m_density_b_prev->length()),
        0);
    
        //diffusion scratch
    blit->fillBuffer(
        m_diffusion_scratch_r,
        NS::Range::Make(0, m_diffusion_scratch_r->length()),
        0);

    blit->fillBuffer(
        m_diffusion_scratch_g,
        NS::Range::Make(0, m_diffusion_scratch_g->length()),
        0);

    blit->fillBuffer(
        m_diffusion_scratch_b,
        NS::Range::Make(0, m_diffusion_scratch_b->length()),
        0);
    

    blit->endEncoding();

    commandBuffer->commit();
    
};


// Metal kernels
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



    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);


    MTL::Size threadgroupSize(16, 16, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeDiffuseDensity(MTL::ComputeCommandEncoder* encoder, float dt, float denom){
   
    encoder->setComputePipelineState(m_diffuseDensityKernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);
    encoder->setBuffer(m_density_r_prev, 0, 3);
    encoder->setBuffer(m_density_g_prev, 0, 4);
    encoder->setBuffer(m_density_b_prev, 0, 5);
    encoder->setBuffer(m_diffusion_scratch_r, 0, 6);
    encoder->setBuffer(m_diffusion_scratch_g, 0, 7);
    encoder->setBuffer(m_diffusion_scratch_b, 0, 8);


    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);

    encoder->setBytes(&dt,
        sizeof(dt),
        9);
    encoder->setBytes(&m_diff_co,
        sizeof(m_diff_co),
        10);

    encoder->setBytes(&width,
        sizeof(width),
        11);
    encoder->setBytes(&height,
        sizeof(height),
        12);
    encoder->setBytes(&denom,
        sizeof(denom),
        13);



    encoder->setBytes(&total,
        sizeof(total),
        14);




    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);


    MTL::Size threadgroupSize(16, 16, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};
void MetalGrid::encodeCopyDensity(MTL::ComputeCommandEncoder* encoder){
   
    encoder->setComputePipelineState(m_copyDensityKernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);
    encoder->setBuffer(m_density_r_prev, 0, 3);
    encoder->setBuffer(m_density_g_prev, 0, 4);
    encoder->setBuffer(m_density_b_prev, 0, 5);
    
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);

    encoder->setBytes(&total,
        sizeof(total),
        6);




    //set number of threads for kernel
    MTL::Size gridSize(m_cellcount, 1, 1);


    MTL::Size threadgroupSize(256, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};


void MetalGrid::encodeComputeDivergence(MTL::ComputeCommandEncoder* encoder){
    //load the kernel
    encoder->setComputePipelineState(m_divergenceKernel);

    //bind the buffers
    encoder->setBuffer(m_u_velocity, 0, 0);
    encoder->setBuffer(m_v_velocity, 0, 1);
    encoder->setBuffer(m_divergence, 0, 2);

    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);

    
    encoder->setBytes(&width,sizeof(width),3);
    encoder->setBytes(&height,sizeof(height),4);
    

    encoder->setBytes(&total,
        sizeof(total),
        5);




    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);


    MTL::Size threadgroupSize(256, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
    return;
}

void MetalGrid::encodeSolvePressure(MTL::ComputeCommandEncoder* encoder){
     //load the kernel
    encoder->setComputePipelineState(m_solvePressureKernel);

    //bind the buffers
    encoder->setBuffer(m_pressure, 0, 0);
    encoder->setBuffer(m_pressure_prev, 0, 1);
    encoder->setBuffer(m_divergence, 0, 2);

    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);

    
    encoder->setBytes(&width,sizeof(width),3);
    encoder->setBytes(&height,sizeof(height),4);
    

    encoder->setBytes(&total,
        sizeof(total),
        5);




    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);


    MTL::Size threadgroupSize(256, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
    
    return;
}

void MetalGrid::encodeProjectPressure(MTL::ComputeCommandEncoder* encoder){
     //load the kernel
    encoder->setComputePipelineState(m_projectKernel);

    //bind the buffers
    encoder->setBuffer(m_pressure, 0, 0);
    encoder->setBuffer(m_u_velocity, 0, 1);
    encoder->setBuffer(m_v_velocity, 0, 2);

    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);

    
    encoder->setBytes(&width,sizeof(width),3);
    encoder->setBytes(&height,sizeof(height),4);
    

    encoder->setBytes(&total,
        sizeof(total),
        5);




    //set number of threads for kernel
    MTL::Size gridSize(width, height, 1);


    MTL::Size threadgroupSize(256, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
    
    return;
}
//Boundary Setter Functions

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


void MetalGrid::encodeBoundaryPressure(MTL::ComputeCommandEncoder* encoder){
    encoder->setComputePipelineState(m_boundaryPressureKernel);

    //bind the buffers
    encoder->setBuffer(m_pressure, 0, 0);

    std::uint32_t width = static_cast<std::uint32_t>(m_width);
    std::uint32_t height = static_cast<std::uint32_t>(m_height);
    std::uint32_t total = static_cast<std::uint32_t>(m_cellcount);
    

    encoder->setBytes(&width,
        sizeof(width),
        1);

    encoder->setBytes(&height,
        sizeof(height),
        2);

    encoder->setBytes(&m_cellcount,sizeof(m_cellcount),3);

    std::uint32_t edge_length =  static_cast<std::uint32_t>(std::max(width,height));


    //set number of threads for kernel
    MTL::Size gridSize(edge_length, 1, 1);


    MTL::Size threadgroupSize(256, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
    return;
};




//Helper functions

void MetalGrid::densityDiffusionHelper(MTL::ComputeCommandEncoder* encoder, float dt){

    //diffusion coefficent
    float denom = 1.f + 4.f * m_diff_co * dt;

    encodeCopyDensity(encoder);

    //iteration loop
    for(std::size_t iteration = 0; iteration < m_diffusion_iterations; iteration++){

        encodeDiffuseDensity(encoder, dt, denom);

        std::swap(m_density_r, m_diffusion_scratch_r);
        std::swap(m_density_g, m_diffusion_scratch_g);
        std::swap(m_density_b, m_diffusion_scratch_b);
        encodeBoundaryDensity(encoder);
        
    }

};

void MetalGrid::solvePressureHelper(MTL::ComputeCommandEncoder* encoder){


    //iteration loop
    for(std::size_t iteration = 0; iteration < m_pressure_iter; iteration++){

        encodeSolvePressure(encoder);

        encodeBoundaryPressure(encoder);

        std::swap(m_pressure, m_pressure_prev);
        
    }
    
    std::swap(m_pressure, m_pressure_prev);
};

std::span<const std::uint8_t> MetalGrid::get_pixels() const
{
    auto* pixels =  static_cast<const std::uint8_t*>(
        m_pixels->contents()
    );

    return {pixels, m_width * m_height *4};
}


MetalGrid::~MetalGrid()
{
    if (m_density_r != nullptr) m_density_r->release();
    if (m_density_g != nullptr) m_density_g->release();
    if (m_density_b != nullptr) m_density_b->release();
    if (m_u_velocity != nullptr) m_u_velocity->release();
    if (m_v_velocity != nullptr) m_v_velocity->release();
    if (m_noiseField != nullptr) m_noiseField->release();
    if (m_diffusion_scratch_r != nullptr) m_noiseField->release();
    if (m_diffusion_scratch_g != nullptr) m_noiseField->release();
    if (m_diffusion_scratch_b != nullptr) m_noiseField->release();
    if (m_pressure != nullptr) m_noiseField->release();
    if (m_density_r_prev != nullptr) m_density_r->release();
    if (m_density_g_prev != nullptr) m_density_g->release();
    if (m_density_b_prev != nullptr) m_density_b->release();
    if (m_u_velocity_prev != nullptr) m_u_velocity->release();
    if (m_v_velocity_prev != nullptr) m_v_velocity->release();
    if (m_divergence != nullptr) m_divergence->release();
    if (m_pressure != nullptr) m_pressure->release();
    if (m_pressure_prev != nullptr) m_pressure_prev->release();
    if (m_pixels != nullptr) m_pixels->release();



    std::cout << "Metal grid destroyed and resources released" << std::endl;

}
