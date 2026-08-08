#include <metal_stdlib>
#include "SimplexNoiseCompute.metal"

using namespace metal;



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
    int gridX = int(center_x)+dx;
    int gridY = int(center_y)+dy;


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

        densityR[index] = new_dens_r * decay;
        densityG[index] = new_dens_g * decay;
        densityB[index] = new_dens_b * decay;


    }


kernel void advectVelocity(
    device float* u_velocity            [[buffer(0)]],
    device float* v_velocity            [[buffer(1)]],
    device const float* u_velocity_prev [[buffer(2)]],
    device const float* v_velocity_prev [[buffer(3)]],
    constant float& dt                  [[buffer(4)]],
    constant float& buoyancy             [[buffer(5)]],
    constant uint& cellcount            [[buffer(6)]],
    constant uint& width                [[buffer(7)]],
    constant uint& height               [[buffer(8)]],
    constant float& velDecay            [[buffer(9)]],
    device const float* densityR        [[buffer(10)]],
    device const float* densityG        [[buffer(11)]],
    device const float* densityB        [[buffer(12)]],
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
    
        //index velocity
        float old_u_vel = u_velocity_prev[index];
        float old_v_vel = v_velocity_prev[index];

        //backwards location lookup 
        float new_u_pos = gid.x - old_u_vel*dt;
        float new_v_pos = gid.y - old_v_vel*dt;

        //finding 4 corners for bilinear interpolation, clamping in boundaries
        float x_sample = clamp(new_u_pos,1.0f,width-2.0f);
        float y_sample = clamp(new_v_pos,1.0f,height-2.0f);

        uint x_floor = floor(x_sample);
        uint x_high = x_floor+1;
        uint y_floor = floor(y_sample);
        uint y_high = y_floor+1;

        //finding values at corners
        uint tl_index = x_floor+y_floor*width;
        uint tr_index = x_high+y_floor*width;
        uint bl_index = x_floor+y_high*width;
        uint br_index = x_high+y_high*width;

        float tl_u = u_velocity_prev[tl_index];
        float tr_u = u_velocity_prev[tr_index];
        float bl_u = u_velocity_prev[bl_index];
        float br_u = u_velocity_prev[br_index];
        
        float tl_v = v_velocity_prev[tl_index];
        float tr_v = v_velocity_prev[tr_index];
        float bl_v = v_velocity_prev[bl_index];
        float br_v = v_velocity_prev[br_index];

        //distance from edges
        float dx = x_sample - x_floor;
        float dy = y_sample - y_floor;

        //weighting calculations
        float tl_weight = (1-dx)*(1-dy);
        float tr_weight = dx*(1-dy);
        float bl_weight = (1-dx)*dy;
        float br_weight = dx*dy;

        //calculate density sample
        float densitySample =  max(densityR[index], max(densityG[index],densityB[index]));

        float new_u_vel = (tl_u*tl_weight + tr_u*tr_weight + bl_u*bl_weight + br_u*br_weight) * velDecay;
        float new_v_vel = (tl_v*tl_weight + tr_v*tr_weight + bl_v*bl_weight + br_v*br_weight) * velDecay;
        
        //adding buoyancy
        new_v_vel -= buoyancy * dt ;
        
        u_velocity[index] = new_u_vel;
        v_velocity[index] = new_v_vel;


    }

kernel void computeNoise(
    device float* noise             [[buffer(0)]],
    constant float& time            [[buffer(1)]],
    constant float& frequency       [[buffer(2)]],
    constant float& timeFrequency   [[buffer(3)]],
    constant uint& width            [[buffer(4)]],
    constant uint& height           [[buffer(5)]],
    constant uint& cellcount        [[buffer(6)]],
    uint2 gid                       [[thread_position_in_grid]]){

        if (gid.x >= width || gid.y >= height) {
            return;
        }

        uint index = gid.y * width + gid.x;

        float x = float(gid.x) * frequency;
        float y = float(gid.y) * frequency;
        float z = time * timeFrequency;


        noise[index] = noise3D(x, y, z);
               
    }


kernel void generateVel(
    device const float* noise   [[buffer(0)]],
    device float* u_velocity    [[buffer(1)]],
    device float* v_velocity    [[buffer(2)]],
    device const float* densityR      [[buffer(3)]],
    device const float* densityG      [[buffer(4)]],
    device const float* densityB      [[buffer(5)]],
    constant float& dt          [[buffer(6)]],
    constant float& curlMult    [[buffer(7)]],
    constant uint& width        [[buffer(8)]],
    constant uint& height       [[buffer(9)]],
    constant uint& cellcount       [[buffer(10)]],
    constant float& strength       [[buffer(11)]],
    uint2 gid                   [[thread_position_in_grid]])
{


    //boundary checks
    if((gid.y*width + gid.x) >= cellcount){
        return;
    }

    if (gid.x == 0 || gid.x >= width - 1 ||
        gid.y == 0 || gid.y >= height - 1) {
        return;
    }

    int index = gid.y*width+gid.x;


    float xl = noise[index-1];
    float xr = noise[index+1];
    float yt = noise[index-width];
    float yb = noise[index+width];

    float dx = ((xr-xl)/2)*strength;
    float dy = ((yb-yt)/2)*strength;

    float densitySample =  clamp(max(densityR[index], max(densityG[index],densityB[index])),0.f,1.f);

    u_velocity[index] += (-dy*curlMult*dt);
    v_velocity[index] += (dx*curlMult*dt);
    

}

kernel void boundaryDensity(
    device float* densityR  [[buffer(0)]],
    device float* densityG  [[buffer(1)]],
    device float* densityB  [[buffer(2)]],
    constant uint& width    [[buffer(3)]],
    constant uint& height   [[buffer(4)]],
    constant uint& cellcount [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
){
    
    if(gid<width){

        int index_top = gid;
        int index_bottom = cellcount - width + gid;

        densityR[index_top] = densityR[index_top+width];
        densityG[index_top] = densityG[index_top+width];
        densityB[index_top] = densityB[index_top+width];
        
        densityR[index_bottom] = densityR[index_bottom-width];
        densityG[index_bottom] = densityG[index_bottom-width];
        densityB[index_bottom] = densityB[index_bottom-width];
    }

    if(gid<height){
        int index_left = gid*width;
        int index_right = (width-1) + gid*width;

        densityR[index_left] = densityR[index_left+1];
        densityG[index_left] = densityG[index_left+1];
        densityB[index_left] = densityB[index_left+1];
        
        densityR[index_right] = densityR[index_right-1];
        densityG[index_right] = densityG[index_right-1];
        densityB[index_right] = densityB[index_right-1];
    }
    
}

kernel void boundaryPressure(
    device float* pressure  [[buffer(0)]],
    constant uint& width    [[buffer(1)]],
    constant uint& height   [[buffer(2)]],
    constant uint& cellcount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
){
    
    if(gid<width){

        int index_top = gid;
        int index_bottom = cellcount - width + gid;

       pressure[index_top] = pressure[index_top+width];
       
       pressure[index_bottom] = pressure[index_bottom-width];
    }

    if(gid<height){
        int index_left = gid*width;
        int index_right = (width-1) + gid*width;

       pressure[index_left] =pressure[index_left+1];
       
       pressure[index_right] =pressure[index_right-1];
    }
    
}
kernel void boundaryVelocity(
    device float* u_velocity  [[buffer(0)]],
    device float* v_velocity  [[buffer(1)]],
    constant uint& width    [[buffer(2)]],
    constant uint& height   [[buffer(3)]],
    constant uint& cellcount [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
){
    
    if(gid<width){

        int index_top = gid;
        int index_bottom = cellcount - width + gid;

        u_velocity[index_top] = u_velocity[index_top+width];
        v_velocity[index_top] = 0.f;
        
        u_velocity[index_bottom] = u_velocity[index_bottom-width];
        v_velocity[index_bottom] = 0.f;

    }

    if(gid<height){
        int index_left = gid*width;
        int index_right = (width-1) + gid*width;

        u_velocity[index_left] = 0.f;
        v_velocity[index_left] = v_velocity[index_left+1];
        
        u_velocity[index_right] = 0.f;
        v_velocity[index_right] = v_velocity[index_right-1];

    }
    
}


kernel void diffuseVelocity(
    device float* u_velocity [[buffer(0)]],
    device float* v_velocity [[buffer(1)]],
    device const float* u_velocity_prev [[buffer(2)]],
    device const float* v_velocity_prev [[buffer(3)]],
    constant float& dt [[buffer(4)]],
    constant float& viscosity [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& height [[buffer(7)]],
    constant uint& cellcount [[buffer(8)]],
    uint2 gid               [[thread_position_in_grid]]
){
    //boundary checks
    if((gid.y*width + gid.x) >= cellcount){
        return;
    }

    if (gid.x == 0 || gid.x >= width - 1 ||
        gid.y == 0 || gid.y >= height - 1) {
        return;
    }

    int index = gid.y*width+gid.x;

   //calc corner indices

    uint l_index = index-1;
    uint r_index = index+1;
    uint u_index = index-width;
    uint d_index = index+width;

    //getting data for laplacian
    float left_u = u_velocity_prev[l_index];
    float right_u = u_velocity_prev[r_index];
    float up_u = u_velocity_prev[u_index];
    float down_u = u_velocity_prev[d_index];
    float center_u = u_velocity_prev[index];

    float left_v = v_velocity_prev[l_index];
    float right_v = v_velocity_prev[r_index];
    float up_v = v_velocity_prev[u_index];
    float down_v = v_velocity_prev[d_index];
    float center_v = v_velocity_prev[index];

    //calculate laplacian
    float lap_u = left_u+right_u+up_u+down_u - (4*center_u);
    float lap_v = left_v+right_v+up_v+down_v - (4*center_v);

    //calculate and write new density
    float new_u_vel = u_velocity_prev[index]+ viscosity * lap_u*dt;
    float new_v_vel = v_velocity_prev[index]+ viscosity * lap_v*dt;

    u_velocity[index] = new_u_vel;
    v_velocity[index] = new_v_vel;




}

kernel void diffuseDensity(
    device float* densityR [[buffer(0)]],
    device float* densityG [[buffer(1)]],
    device float* densityB [[buffer(2)]],
    device const float* densityR_prev [[buffer(3)]],
    device const float* densityG_prev [[buffer(4)]],
    device const float* densityB_prev [[buffer(5)]],
    device float* scratch_R [[buffer(6)]],
    device float* scratch_G [[buffer(7)]],
    device float* scratch_B [[buffer(8)]],
    constant float& dt [[buffer(9)]],
    constant float& diff_co [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    constant uint& height [[buffer(12)]],
    constant float& denom [[buffer(13)]],
    constant uint& cellcount [[buffer(14)]],
    uint2 gid               [[thread_position_in_grid]]
    ){

    //boundary checks
    if((gid.y*width + gid.x) >= cellcount){
        return;
    }

    if (gid.x == 0 || gid.x >= width - 1 ||
        gid.y == 0 || gid.y >= height - 1) {
        return;
    }
    //iteration setup


    uint index = gid.y*width + gid.x;

    float neighbourSum_R = 
        densityR[index-1] +
        densityR[index+1] + 
        densityR[index-width] + 
        densityR[index+width];

    float neighbourSum_G = 
        densityG[index-1] +
        densityG[index+1] + 
        densityG[index-width] + 
        densityG[index+width];

    float neighbourSum_B = 
        densityB[index-1] +
        densityB[index+1] + 
        densityB[index-width] + 
        densityB[index+width];

    scratch_R[index] = 
    (densityR_prev[index] + dt * diff_co * neighbourSum_R)/ denom;

    scratch_G[index] = 
    (densityG_prev[index] + dt * diff_co * neighbourSum_G)/ denom;

    scratch_B[index] = 
    (densityB_prev[index] + dt * diff_co * neighbourSum_B)/ denom;
        
}


kernel void copyDensity(
    device float* densityR [[buffer(0)]],
    device float* densityG [[buffer(1)]],
    device float* densityB [[buffer(2)]],
    device const float* densityR_prev [[buffer(3)]],
    device const float* densityG_prev [[buffer(4)]],
    device const float* densityB_prev [[buffer(5)]],
    constant uint& cellcount [[buffer(6)]],
    uint gid                [[thread_position_in_grid]]
){
    if(gid>=cellcount){
        return;
    }
    densityR[gid] = densityR_prev[gid];
    densityG[gid] = densityG_prev[gid];
    densityB[gid] = densityB_prev[gid];
}


kernel void computeDivergence(
    device const float* u_velocity [[buffer(0)]],
    device const float* v_velocity [[buffer(1)]],
    device float* divergence [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& cellcount [[buffer(5)]],
    uint2 gid                [[thread_position_in_grid]]
){
    if(gid.y*width+gid.x>=cellcount){
        return;
    }

    if(gid.x ==0 || gid.x == width-1 || gid.y == 0 || gid.y == height-1){
        return;
    }

    uint index = gid.y*width+gid.x;

    float dx = (u_velocity[index+1] - u_velocity[index-1]) *.5;
    float dy = (v_velocity[index+width] - v_velocity[index-width]) *.5;

    divergence[index] = dx+dy;

}

kernel void solvePressure(
    device float* pressure [[buffer(0)]],
    device const float* pressure_prev [[buffer(1)]],
    device const float* divergence [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& cellcount [[buffer(5)]],
    uint2 gid                [[thread_position_in_grid]]
){
    if(gid.y*width+gid.x>=cellcount){
        return;
    }

    if(gid.x ==0 || gid.x == width-1 || gid.y == 0 || gid.y == height-1){
        return;
    }

    uint index = gid.y*width+gid.x;

    float xl = pressure_prev[index-1];
    float xr = pressure_prev[index+1];
    float yt = pressure_prev[index+width]; 
    float yb = pressure_prev[index-width];

    pressure[index] = (xl+xr+yt+yb - divergence[index])*0.25f;


}

kernel void project(
    device const float* pressure [[buffer(0)]],
    device float* u_velocity [[buffer(1)]],
    device float* v_velocity [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& cellcount [[buffer(5)]],
    uint2 gid                [[thread_position_in_grid]]
){
    if(gid.y*width+gid.x>=cellcount){
        return;
    }

    if(gid.x ==0 || gid.x == width-1 || gid.y == 0 || gid.y == height-1){
        return;
    }

    uint index = gid.y*width+gid.x;

    float xl = pressure[index-1];
    float xr = pressure[index+1];
    float yt = pressure[index+width]; 
    float yb = pressure[index-width];
    
    float u_project = (xr-xl)/2;
    float v_project = (yt-yb)/2;

    u_velocity[index] -= u_project;
    v_velocity[index] -= v_project;

}
