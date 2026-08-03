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
    m_buoyancy(50.f),
    m_diff_co(4.5f),
    m_viscosity(0.05f),
    m_decay(0.994f),
    m_curl_mult(1000),
    m_noise_freq(0.04f),


    m_seed(seed),
    m_width(width),
    m_height(height),
    m_cellcount(cell_count),
    m_bytesize(bytesize),
    m_metalcontext(MetalContext),

    m_vel_decay(0.9f),
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

    m_diffusion_scratch(width*height,0.f),
    m_diffusion_iterations{20},

    m_buoyancyKernel(nullptr),
    m_densityDecaykernel(nullptr),
    m_emitterKernel(nullptr),
    m_pixelKernel(nullptr),

    m_density_r_prev(nullptr),
    m_density_g_prev(nullptr),
    m_density_b_prev(nullptr),
    m_u_velocity_prev(nullptr),
    m_v_velocity_prev(nullptr),

    m_noise(width*height, 0.0f),
    m_pressure(width*height, 0.0f),
    m_pressure_prev(width*height, 0.0f),
    m_divergence(width*height, 0.0f)
    {
        //create error object
         NS::Error* error = nullptr;

        // configure reproducible noise generator
        m_noise_gen.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        m_noise_gen.SetSeed(seed);

        //load compute pipeline kernels
        m_buoyancyKernel = m_metalcontext.CreatePipelineState("buoyancy", error);
    
        if(m_buoyancyKernel == nullptr){
            std::cerr << "problem creating buoyancy kernel" << std::endl;
            return; 
        }

        m_densityDecaykernel = m_metalcontext.CreatePipelineState("density_decay", error);
    
        if(m_densityDecaykernel == nullptr){
            std::cerr << "problem creating density decay kernel" << std::endl;
            return; 
        }

        m_emitterKernel = m_metalcontext.CreatePipelineState("emitter", error);
    
        if(m_emitterKernel == nullptr){
            std::cerr << "problem creating emitter kernel" << std::endl;
            return; 
        }
        
        m_pixelKernel = m_metalcontext.CreatePipelineState("gen_pixels", error);
    
        if(m_pixelKernel == nullptr){
            std::cerr << "problem creating gen pixels kernel" << std::endl;
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
        m_pixels = m_metalcontext.get_device()->newBuffer(
            m_bytesize*4.f,
            MTL::ResourceStorageModeShared
        );

        
        // if any error in creating the buffers: printout error, call destructor, shut it down
        if (m_density_r == nullptr ||
            m_density_g == nullptr || 
            m_density_b == nullptr ||
            m_v_velocity == nullptr ||
            m_u_velocity == nullptr ||
            m_pixels == nullptr) {
                
                std::cerr << "Error allocating Metal buffers.\n";
                
                //if buffer didn't err out release memory before end
                if (m_density_r != nullptr) m_density_r->release();
                if (m_density_g != nullptr) m_density_g->release();
                if (m_density_b != nullptr) m_density_b->release();
                if (m_u_velocity != nullptr) m_u_velocity->release();
                if (m_v_velocity != nullptr) m_v_velocity->release();
                if (m_pixels != nullptr) m_v_velocity->release();
                
                return;
            }
            
        //initialize buffer values
        float* r = static_cast<float*>(m_density_r->contents());
        float* g = static_cast<float*>(m_density_g->contents());
        float* b = static_cast<float*>(m_density_b->contents());
        float* u = static_cast<float*>(m_u_velocity->contents());
        float* v = static_cast<float*>(m_v_velocity->contents());
        float* p = static_cast<float*>(m_pixels->contents());

        std::fill_n(r,m_cellcount,0.f);
        std::fill_n(g,m_cellcount,0.f);
        std::fill_n(b,m_cellcount,0.f);
        std::fill_n(u,m_cellcount,0.f);
        std::fill_n(v,m_cellcount,0.f);
        std::fill_n(p,m_cellcount,0.f);


    };

void MetalGrid::update(float dt){

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

    //encode emitter kernel
    for(auto& emitter : m_emitters){
        encodeEmitter(encoder, emitter);
    }
    //encode buoyancy kernel
    encodeBuoyancy(encoder, dt);
    //encode decay kernel
    encodeDecay(encoder);

    //encode pixels
    encodePixels(encoder);




    //end encoding for kernel
    encoder->endEncoding();

    //submit the work and wait for completion.

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

};

void MetalGrid::encodeBuoyancy(MTL::ComputeCommandEncoder* encoder, float dt){
    //set the buoyancy kernel to the encoder
    encoder->setComputePipelineState(m_buoyancyKernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);
    encoder->setBuffer(m_v_velocity, 0, 3);


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

    //set number of threads for kernel
    MTL::Size gridSize(m_cellcount, 1, 1);

    const std::size_t threadgroupWidth =
        std::min(
            m_buoyancyKernel->maxTotalThreadsPerThreadgroup(),
            m_cellcount
        );

    MTL::Size threadgroupSize(threadgroupWidth, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeDecay(MTL::ComputeCommandEncoder* encoder){
  //set the buoyancy kernel to the encoder
    encoder->setComputePipelineState(m_densityDecaykernel);

    //bind the buffers
    encoder->setBuffer(m_density_r, 0, 0);
    encoder->setBuffer(m_density_g, 0, 1);
    encoder->setBuffer(m_density_b, 0, 2);


    encoder->setBytes(
        &m_decay,
        sizeof(float),
        3
    );

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
            m_densityDecaykernel->maxTotalThreadsPerThreadgroup(),
            m_cellcount
        );

    MTL::Size threadgroupSize(threadgroupWidth, 1, 1);

    encoder->dispatchThreads(
        gridSize,
        threadgroupSize
    );
};

void MetalGrid::encodeEmitter(MTL::ComputeCommandEncoder* encoder, Emitter* emitter){
      //set the buoyancy kernel to the encoder
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


void MetalGrid::encodePixels(MTL::ComputeCommandEncoder* encoder){
    //set the buoyancy kernel to the encoder
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

const std::uint8_t* MetalGrid::get_pixels() const
{
    return static_cast<const std::uint8_t*>(
        m_pixels->contents()
    );
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

    if(m_buoyancyKernel != nullptr){
        m_buoyancyKernel->release();
        m_buoyancyKernel = nullptr;
    };

    std::cout << "Metal grid destroyed and resources released" << std::endl;

}
