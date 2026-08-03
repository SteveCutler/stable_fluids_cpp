#include "MetalContext.hpp"
#include <iostream>

//Metal Context creates and owns the key persisting metal objects: device, command queue and library

MetalContext::MetalContext(std::string library):
m_library_name(library){

    //Intialize metal device

    m_device = MTL::CreateSystemDefaultDevice();


    if (m_device == nullptr) {
        std::cerr << "Error: no Metal device found.\n";
        return;
    }

     std::cout << "Metal device: "
              << m_device->name()->utf8String()
              << '\n';


    //Create command queue

    m_commandqueue =
        m_device->newCommandQueue();

    if (m_commandqueue == nullptr) {
        std::cerr << "Error creating command queue.\n";
        return;
    }
    
    //Load metal library

    NS::Error* error = nullptr;

    NS::String* libraryPath =
        NS::String::string(
            m_library_name.c_str(),
            NS::UTF8StringEncoding
        );

    NS::URL* libraryURL =
        NS::URL::fileURLWithPath(libraryPath);

    MTL::Library* m_library =
        m_device->newLibrary(libraryURL, &error);

    if (m_library == nullptr) {
        std::cerr << "Error loading: " << library << std::endl;

        if (error != nullptr) {
            std::cerr << ": "
                      << error->localizedDescription()->utf8String();
        }

        std::cerr << '\n';

        m_device->release();
        return;
    }
    
}

//method to turn the metal functions into compute pipeline states that get encoded

MTL::ComputePipelineState* MetalContext::CreatePipelineState(std::string kernel_name, NS::Error* error){

    //search for metal function name in library
    MTL::Function* function = m_library->newFunction(
        NS::String::string(
            kernel_name.c_str(),
            NS::UTF8StringEncoding
        )
    );

    if(function == nullptr){
        std::cerr << "Error: kernel " << kernel_name << " was not found. \n";
        
        m_library->release();
        m_device->release();
        return nullptr;
    }

    //create pipeline state out of function
    MTL::ComputePipelineState* pipeline = 
        m_device->newComputePipelineState(function, &error);
    
        if(pipeline == nullptr){
            std::cerr << "Error creating compute pipeline";

            if(error != nullptr){
                std::cerr << ": " << error->localizedDescription()->utf8String() << "\n";
            }

            function->release();
           
            return nullptr;
        }

    function->release();
    return pipeline;
}

MetalContext::~MetalContext(){

    if(m_library != nullptr){
        m_library->release();
    }
    if(m_commandqueue != nullptr){
        m_commandqueue->release();
    }
    if(m_device != nullptr){
        m_device->release();
    }

    std::cout << "Metal Context destroyed" << std::endl;
}

//getters
MTL::Device* MetalContext::get_device(){
    return m_device;
}
MTL::CommandQueue* MetalContext::get_commandqueue(){
    return m_commandqueue;
}
