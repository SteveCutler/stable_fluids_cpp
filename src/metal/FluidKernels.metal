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
    int dx = int(local.x) - radius;
    int dy = int(local.y) - radius;

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
    device uchar4* pixels         [[buffer(3)]],
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
    uchar value_r = uchar(clamp(d_r,0.f,1.f) * 255.f);
    uchar value_g = uchar(clamp(d_g,0.f,1.f) * 255.f);
    uchar value_b = uchar(clamp(d_b,0.f,1.f) * 255.f);
    
    //find correct position in pixel array, given each pixel has 1 components
    int p = index;
    
    //create greyscale image with alpha of 1
    pixels[p] = uchar4(value_r, value_g, value_b, uchar(255)); 

}


kernel void advectDensity(
    device float* densityR              [[buffer(0)]],
    device float* densityG              [[buffer(1)]],
    device float* densityB              [[buffer(2)]],
    device const float* densityR_prev   [[buffer(3)]],
    device const float* densityG_prev   [[buffer(4)]],
    device const float* densityB_prev   [[buffer(5)]],
    device const float* u_velocity      [[buffer(6)]],
    device const float* v_velocity      [[buffer(7)]],
    constant float& dt                  [[buffer(8)]],
    constant uint& cellcount            [[buffer(9)]],
    constant uint& width                [[buffer(10)]],
    constant uint& height               [[buffer(11)]],
    constant float& decay               [[buffer(12)]],
    uint2 gid                           [[thread_position_in_grid]]){


        //boundary checks
        if((gid.y*width + gid.x) >= cellcount){
            return;
        }

        if (gid.x == 0 || gid.x >= width - 1 ||
            gid.y == 0 || gid.y >= height - 1) {
            return;
        }

        // calculate index
        int index = gid.y*width + gid.x;

        //get vel
        float u_vel = u_velocity[index];
        float v_vel = v_velocity[index];


        //backwards location lookup 
        float new_x = clamp((gid.x - u_vel*dt),1.f,width-2.f);
        float new_y = clamp((gid.y - v_vel*dt),1.f,height-2.f);

        //finding 4 corners for bilinear interpolation, clamping in boundaries
        uint x_low = floor(new_x);
        uint x_high = x_low+1;
        uint y_low = floor(new_y);
        uint y_high = y_low+1;

        //distance from edges
        float dx = new_x - x_low;
        float dy = new_y - y_low;

        //weighting calculations
        float tl_weight = (1-dx)*(1-dy);
        float tr_weight = dx*(1-dy);
        float bl_weight = (1-dx)*dy;
        float br_weight = dx*dy;

        //sampling values at corners
        float tl_r = densityR_prev[x_low+y_low*width];
        float tr_r = densityR_prev[x_high+y_low*width];
        float bl_r = densityR_prev[x_low+y_high*width];
        float br_r = densityR_prev[x_high+y_high*width];
        
        float tl_g = densityG_prev[x_low+y_low*width];
        float tr_g = densityG_prev[x_high+y_low*width];
        float bl_g = densityG_prev[x_low+y_high*width];
        float br_g = densityG_prev[x_high+y_high*width];
        
        float tl_b = densityB_prev[x_low+y_low*width];
        float tr_b = densityB_prev[x_high+y_low*width];
        float bl_b = densityB_prev[x_low+y_high*width];
        float br_b = densityB_prev[x_high+y_high*width];


        float new_dens_r = tl_r*tl_weight + tr_r*tr_weight + bl_r*bl_weight + br_r*br_weight;
        float new_dens_g = tl_g*tl_weight + tr_g*tr_weight + bl_g*bl_weight + br_g*br_weight;
        float new_dens_b = tl_b*tl_weight + tr_b*tr_weight + bl_b*bl_weight + br_b*br_weight;

        densityR[index] = new_dens_r*decay;
        densityG[index] = new_dens_g*decay;
        densityB[index] = new_dens_b*decay;


    }

