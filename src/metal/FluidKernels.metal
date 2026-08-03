#include <metal_stdlib>
using namespace metal;

kernel void buoyancy(
    device const float* densityR [[buffer(0)]],
    device const float* densityG [[buffer(1)]],
    device const float* densityB [[buffer(2)]],
    device float* v_velocity     [[buffer(3)]],
    constant float& dt           [[buffer(4)]],
    constant float& buoyancy     [[buffer(5)]],
    constant uint& cellcount     [[buffer(6)]],
    uint index                   [[thread_position_in_grid]])
{

    //out of bound check
    if(index >= cellcount){
        return;
    }

    //sample max density value
    float value = max(
        densityR[index],
            max(
                densityG[index],
                densityB[index])
    );
    
    v_velocity[index] -= buoyancy * clamp(value, 0.f, 1.f) * dt ;
}

kernel void density_decay(
    device float* densityR      [[buffer(0)]],
    device float* densityG      [[buffer(1)]],
    device float* densityB      [[buffer(2)]],
    constant float& decay       [[buffer(3)]],
    constant uint& cellcount    [[buffer(4)]],
    uint index                  [[thread_position_in_grid]])
{
    //out of bound check
    if(index >= cellcount){
        return;
    }

    densityR[index] *= decay;
    densityG[index] *= decay;
    densityB[index] *= decay;
}

kernel void  emitter(
    device float* densityR      [[buffer(0)]],
    device float* densityG      [[buffer(1)]],
    device float* densityB      [[buffer(2)]],
    constant uint& center_x     [[buffer(3)]],
    constant uint& center_y     [[buffer(4)]],
    constant float& radius      [[buffer(5)]],
    constant float& r           [[buffer(6)]],
    constant float& g           [[buffer(7)]],
    constant float& b           [[buffer(8)]],
    constant uint& bound        [[buffer(9)]],
    constant uint& width        [[buffer(10)]],
    constant uint& height       [[buffer(11)]],
    uint2 local                 [[thread_position_in_grid]])
    {     

    //turn the local coordinates into offset that can be applied around the center of the emitter
    int dx = int(local.x) - bound;
    int dy = int(local.y) - bound;

    //actualy grid coordinates
    int gridX = center_x+dx;
    int gridY = center_y+dy;


    //out of boundary check
    if(gridX<=1 || gridY<=1 ||
    gridX>=int(width)-1 || gridY>=int(height)-1){
        return;
    }

    //distance check to round the square into a circle
    float distance_sqrd = float(dx*dx+dy*dy);
    float rad_sqr = radius*radius;

    if(rad_sqr <=0.f || distance_sqrd >= rad_sqr){
        return;
    }

    //calculate index
    uint index = gridX + gridY*width;

    
    float red = r * (rad_sqr-distance_sqrd)/rad_sqr;
    float green = g * (rad_sqr-distance_sqrd)/rad_sqr;
    float blue = b * (rad_sqr-distance_sqrd)/rad_sqr;

    
    densityR[index] = clamp((densityR[index]+red),0.f,1.f);
    densityG[index] = clamp((densityG[index]+green),0.f,1.f);
    densityB[index] = clamp((densityB[index]+blue),0.f,1.f);
                
}

kernel void gen_pixels(
    device const float* densityR [[buffer(0)]],
    device const float* densityG [[buffer(1)]],
    device const float* densityB [[buffer(2)]],
    device float* pixels         [[buffer(3)]],
    device const uint& cellcount [[buffer(4)]],
    uint index                   [[thread_position_in_grid]])
{
    if(index >= cellcount){
        return;
    }

    float d_r = densityR[index];
    float d_g = densityG[index];
    float d_b = densityB[index];
    
    
    //convert density to RGB values
    uint value_r = uint(d_r * 255.f);
    uint value_g = uint(d_g * 255.f);
    uint value_b = uint(d_b * 255.f);
    
    //find correct position in pixel array, given each pixel has 1 components
    int p = index * 4;
    
    //create greyscale image with alpha of 1
    pixels[p] = value_r;  //R
    pixels[p+1] = value_g;//G
    pixels[p+2] = value_b;//B
    pixels[p+3] = 255;  //Alpha channel
}