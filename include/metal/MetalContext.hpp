#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

class MetalContext {

    public:
        //constructor
        MetalContext(std::string library);
        
        //Create kernel pipeline
        MTL::ComputePipelineState* CreatePipelineState(const std::string& kernel);
        
        //destructor
        ~MetalContext();

        //getters
        MTL::Device* get_device();
        MTL::CommandQueue* get_commandqueue();
        
        std::string m_library_name;

        //delete copying
        MetalContext(const MetalContext&) = delete;
        MetalContext& operator=(const MetalContext&) = delete;

    private:
        MTL::Device* m_device = nullptr;
        MTL::CommandQueue* m_commandqueue = nullptr;
        MTL::Library* m_library = nullptr;


};