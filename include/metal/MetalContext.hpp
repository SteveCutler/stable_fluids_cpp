#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

class MetalContext {

    public:
        //constructor
        MetalContext(std::string library);
        
        //Create kernel pipeline
        MTL::ComputePipelineState* CreatePipelineState(std::string kernel_name, NS::Error* error);
        
        //destructor
        ~MetalContext();

        //getters
        MTL::Device* get_device();
        MTL::CommandQueue* get_commandqueue();
        
        std::string m_library_name;

    private:
        MTL::Device* m_device;
        MTL::CommandQueue* m_commandqueue;
        MTL::Library* m_library;


};