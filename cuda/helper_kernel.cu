#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include <ATen/ATen.h>
#include <stdint.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

#include <torch/extension.h>
#include "interpolation.h"
#include "dda.h"
#include "grid.h"
#include "camera.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
typedef unsigned long long int uint64_cu;

__device__ 
inline float3 fetch_color_forward(uint8_t* src, float2 uv, int height, int width, bool &valid)
{
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    if (uv.x < 0 || uv.x >= width || uv.y < 0 || uv.y >= height)
    {
        valid = false;
        return color;
    }
    valid = true;

    float total_weight = 0.0f; 
    uv.x = uv.x - 0.5f;
    uv.y = uv.y - 0.5f;
    int u = (int)floorf(uv.x);
    int v = (int)floorf(uv.y);

    float2 w_base = make_float2(uv.x - u, uv.y - v);
    float w[4] = {  
        (1.0f - w_base.x) * (1.0f - w_base.y),
        (1.0f - w_base.x) * w_base.y,
        w_base.x * (1.0f - w_base.y),
        w_base.x * w_base.y};

    for(int i=0; i<2; i++)
    {
        for(int j=0; j<2; j++)
        {
            int x = u + i;
            int y = v + j;
            if (x < 0 || y < 0 || x >= width || y >= height) continue;

            float weight = w[i*2+j];
            total_weight += weight;
            int idx = y * width + x;
            color.x += weight * (float)src[idx * 3 + 0];
            color.y += weight * (float)src[idx * 3 + 1];
            color.z += weight * (float)src[idx * 3 + 2];
        }
    }

    if (total_weight > 0.0f)
    {
        color = color / total_weight;
    }
    return color;
}

__device__ 
inline float2 fetch_color_backward(uint8_t* src, float2 uv, int height, int width, 
                                   float3 grid_in)
{
    float2 out = make_float2(0,0);
    if (uv.x < 0 || uv.x >= width || uv.y < 0 || uv.y >= height)
    {
        return out;
    }

    float total_weight = 0.0f; 
    uv.x = uv.x - 0.5f;
    uv.y = uv.y - 0.5f;
    int u = (int)floorf(uv.x);
    int v = (int)floorf(uv.y);

    float2 w_base = make_float2(uv.x - u, uv.y - v);

    float w[4] = {  
        (1.0f - w_base.x) * (1.0f - w_base.y),
        (1.0f - w_base.x) * w_base.y,
        w_base.x * (1.0f - w_base.y),
        w_base.x * w_base.y};

    float dwdu[4] = {-1.0f * (1.0f - w_base.y),
                     -1.0f * w_base.y, 
                     1.0f - w_base.y, w_base.y};
    float dwdv[4] = {-1.0f * (1.0f - w_base.x),
                     1.0f - w_base.x,
                     -1.0f * w_base.x, w_base.x};
    
    for(int i=0; i<2; i++)
    {
        for(int j=0; j<2; j++)
        {
            int x = u + i;
            int y = v + j;
            if (x < 0 || y < 0 || x >= width || y >= height) continue;

            int idx = y * width + x;

            float dL_dw = grid_in.x * (float)src[idx * 3 + 0] + 
                          grid_in.y * (float)src[idx * 3 + 1] + 
                          grid_in.z * (float)src[idx * 3 + 2];

            out.x += dL_dw * dwdu[i*2+j];
            out.y += dL_dw * dwdv[i*2+j];
            total_weight += w[i*2+j];
            
        }
    } 
    // [FIXME]
    if (total_weight > 0.0f)
    {
        out = out / total_weight;
    }
    return out;
}


__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__device__ 
float e_ibr(float3 sample, float3 oi, float3 oj)
{
    // oi is reference camera center
    float3 vi = oi - sample;
    float3 vj = oj - sample;

    float angle_cost = fminf(1.0f - dot(normalize(vi), normalize(vj)), 1.0f);

    // [FIXME] 选择距离差不多的比较合理
    float dis_cost = fmaxf(0.0f, 1.0f - norm(vi)/norm(vj));

    return 0.9f * angle_cost + 0.1f * dis_cost;
}


__device__ 
float angle_weight(float3 sample, float3 oi, float3 oj)
{
    // oi is reference camera center
    float3 vi = oi - sample;
    float3 vj = oj - sample;
    return 1.0f - fminf(fmaxf(norm(cross(normalize(vi), normalize(vj))), 0.0f),1.0f);
}



__global__
void proj2pixel_and_fetch_color_kernel(
    float3* pts, // B 
    float* Ks, // N x 9
    float* C2Ws, // N x 12
    float3* RGBs, // N x H x W  
    float3* fetched_pixels, // B x N 
    float3* fetched_colors, // B x N 每个点从每个view拿到的颜色
    int height, int width)
{
    // blockDim.x = 512  gridDim.x = batch_size, gridDim.y = num_camera 
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < gridDim.x)
    {
        float* K = Ks + blockIdx.y * 9;
        float* C2W = C2Ws + blockIdx.y * 12;
        float3* RGB = RGBs + blockIdx.y * height * width;
        float3* fetched_rgb = fetched_colors + taskIdx * gridDim.y + blockIdx.y;
        float3* fetched_loc = fetched_pixels + taskIdx * gridDim.y + blockIdx.y;

        float3 p = pts[taskIdx];

        p.x -= C2W[3];
        p.y -= C2W[7];
        p.z -= C2W[11];

        float x_cam = C2W[0] * p.x + C2W[4] * p.y + C2W[8] * p.z;
        float y_cam = C2W[1] * p.x + C2W[5] * p.y + C2W[9] * p.z;
        float z_cam = C2W[2] * p.x + C2W[6] * p.y + C2W[10] * p.z;

        float pixel_x = K[0] * x_cam + K[1] * y_cam + K[2] * z_cam;
        float pixel_y = K[3] * x_cam + K[4] * y_cam + K[5] * z_cam;
        float pixel_z = K[6] * x_cam + K[7] * y_cam + K[8] * z_cam;

        if (pixel_z > 0)
        {
            pixel_x /= pixel_z;
            pixel_y /= pixel_z;

            if (pixel_x >= 0 && pixel_x <= width-1 && pixel_y >=0 && pixel_y <= height-1)
            {
                Bilinear<float, 3>((float*)RGB, (float*)fetched_rgb, height, width, 
                                    make_float2(pixel_x, pixel_y));
                fetched_loc[0] = make_float3(pixel_x, pixel_y, z_cam);
            }else{
                fetched_rgb[0] = make_float3(0.0f, 0.0f, 0.0f);
                fetched_loc[0] = make_float3(-1.0f, -1.0f, -1.0f);
            }

            
        }else{
            fetched_rgb[0] = make_float3(0.0f, 0.0f, 0.0f);
            fetched_loc[0] = make_float3(-1.0f, -1.0f, -1.0f);
        }

        taskIdx += total_thread;
    }
}

void proj2pixel_and_fetch_color(
    at::Tensor pts,
    at::Tensor Ks, 
    at::Tensor C2Ws,
    at::Tensor RGBs,
    at::Tensor &fetched_pixels,
    at::Tensor &fetched_colors)
{
    int batch_size = pts.size(0);
    int height = RGBs.size(1);
    int width = RGBs.size(2);
    int num_camera = Ks.size(0);

    dim3 num_block = dim3(batch_size, num_camera);

    proj2pixel_and_fetch_color_kernel<<<num_block, NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(),
        Ks.contiguous().data_ptr<float>(),
        C2Ws.contiguous().data_ptr<float>(),
        (float3*)RGBs.contiguous().data_ptr<float>(),
        (float3*)fetched_pixels.contiguous().data_ptr<float>(),
        (float3*)fetched_colors.contiguous().data_ptr<float>(),
        height, width);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__global__ 
void ray_aabb_intersection_kernel(
    float3* rays_o, 
    float3* rays_d,
    float3* aabb_center,
    float3* aabb_size,
    float2* bounds, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];
        
        bounds[taskIdx] = RayAABBIntersection(origin, direction, aabb_center[0], aabb_size[0] / 2.0f);

        taskIdx += total_thread;
    }
}

void ray_aabb_intersection(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds)
{
    int batch_size = rays_o.size(0);
    ray_aabb_intersection_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)aabb_center.contiguous().data_ptr<float>(),
        (float3*)aabb_size.contiguous().data_ptr<float>(),
        (float2*)bounds.contiguous().data_ptr<float>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}



__global__ 
void ray_aabb_intersection_v2_kernel(
    float3* rays_o,  // B x 3 
    float3* rays_d,  // B x 3 
    float3* aabb_center, //  K x 3
    float3* aabb_size, //  K x 3
    float2* bounds, // B x K 
    int num_aabb, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    int box_idx = blockIdx.y;

    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];
        
        bounds[taskIdx*num_aabb+box_idx] = RayAABBIntersection(origin, direction, aabb_center[box_idx], aabb_size[box_idx] / 2.0f);

        taskIdx += total_thread;
    }
}

void ray_aabb_intersection_v2(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds)
{
    int batch_size = rays_o.size(0);
    int num_aabb = aabb_center.size(0);

    ray_aabb_intersection_v2_kernel<<<dim3(NUM_BLOCK(batch_size), num_aabb), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)aabb_center.contiguous().data_ptr<float>(),
        (float3*)aabb_size.contiguous().data_ptr<float>(),
        (float2*)bounds.contiguous().data_ptr<float>(), num_aabb, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}



__device__ 
inline float3 fetch_color(uint8_t* src, float2 uv, int height, int width,
              bool &valid)
{

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    if (uv.x < 0 || uv.x >= width || uv.y < 0 || uv.y >= height)
    {
        valid = false;
        return color;
    }

    valid = true;

    // 实现0.5,0.5像素中心版本的双线性插值 
    float total_weight = 0.0f; 
    

    uv.x = uv.x - 0.5f;
    uv.y = uv.y - 0.5f;
    int u = (int)floorf(uv.x);
    int v = (int)floorf(uv.y);

    float2 w_base = make_float2(uv.x - u, uv.y - v);
    float w[4] = {  
        (1.0f - w_base.x) * (1.0f - w_base.y),
        (1.0f - w_base.x) * w_base.y,
        w_base.x * (1.0f - w_base.y),
        w_base.x * w_base.y};

    for(int i=0; i<2; i++)
    {
        for(int j=0; j<2; j++)
        {
            int x = u + i;
            int y = v + j;
            if (x < 0 || y < 0 || x >= width || y >= height) continue;

            float weight = w[i*2+j];
            total_weight += weight;
            int idx = y * width + x;
            color.x += weight * (float)src[idx * 3 + 0];
            color.y += weight * (float)src[idx * 3 + 1];
            color.z += weight * (float)src[idx * 3 + 2];
        }
    }

    if (total_weight > 0.0f)
    {
        color = color / total_weight;
    }
    return color;


    // int x0 = (int)uv.x;
    // int y0 = (int)uv.y;
    // int x1 = x0 + 1;
    // int y1 = y0 + 1;
    // float x = uv.x - x0;
    // float y = uv.y - y0;

    // int idx00 = x0 + y0 * width;
    // int idx01 = idx00 + width;
    // int idx10 = idx00 + 1;
    // int idx11 = idx01 + 1;

    // float3 v00 = make_float3((float)src[idx00 * 3 + 0], 
    //                          (float)src[idx00 * 3 + 1],
    //                          (float)src[idx00 * 3 + 2]);
    // float3 v01 = make_float3((float)src[idx01 * 3 + 0], 
    //                          (float)src[idx01 * 3 + 1],
    //                          (float)src[idx01 * 3 + 2]);
    // float3 v10 = make_float3((float)src[idx10 * 3 + 0], 
    //                          (float)src[idx10 * 3 + 1],
    //                          (float)src[idx10 * 3 + 2]);
    // float3 v11 = make_float3((float)src[idx11 * 3 + 0], 
    //                          (float)src[idx11 * 3 + 1],
    //                          (float)src[idx11 * 3 + 2]);
    

    // return v00 * (1.0f - x) * (1.0f - y) + 
    //        v01 * (1.0f - x) * y + 
    //        v10 * x * (1.0f - y) + 
    //        v11 * x * y;
}


__device__ 
inline float3 gaussian_interpolate(uint8_t* src, float2 uv, int height, int width, float sigma, float max_dis)
{
    /*
    sigma  高斯函数的sigma 
    max_dis  最大距离
    */

    float item = -1.0f / (sigma * sigma);

    // // src H x W x 3 
    // uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
    // uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

    if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
    {
        return make_float3(0,0,0);
    }

    int x0 = (int)uv.x;
    int y0 = (int)uv.y;

    int M = (int)(max_dis * 2) + 2; 
    int S = M / 2;

    float total_weight = 0;
    float3 color = make_float3(0,0,0);

    for (int i=0; i<M; i++)
    {
        for (int j=0; j<M; j++)
        {
            int loc_x = x0 + i - S;
            int loc_y = y0 + j - S;

            if(loc_x < 0 || loc_x >= width || loc_y < 0 || loc_y >= height) continue;

            float x = loc_x + 0.5f;
            float y = loc_y + 0.5;

            float dis = (x - uv.x) * (x - uv.x) + (y - uv.y) * (y - uv.y);

            float w = expf(item * dis);

            int index = loc_y * width + loc_x;

            float3 c = make_float3((float)src[index*3+0], (float)src[index*3+1], (float)src[index*3+2]);
            color = color + w * c;

            total_weight += w;
        }
    }
    if (total_weight > 0)
    {
        color = color / total_weight;
    }
    return color;
}


__global__ 
void gen_rays_kernel(
    int* view_idxs, // B
    int* rays_idxs, // B 
    PinholeCameraManager cameras, 
    float3* rays_o, // B x 3
    float3* rays_d, // B x 3 
    float3* up_axis, // B x 3
    int height, int width, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        int vidx = view_idxs[taskIdx];
        int ridx = rays_idxs[taskIdx];
        PinholeCamera cam = cameras[vidx];

        float2 uv = make_float2((float)(ridx % width) + 0.5f, 
                                (float)(ridx / width) + 0.5f);

        float3 x_world = cam.pixel2world(uv, 1.0f);

        float3 origin = cam.camera_center();

        rays_o[taskIdx] = origin;
        rays_d[taskIdx] = x_world - origin;
        up_axis[taskIdx] = make_float3(cam.E.data[4], cam.E.data[5], cam.E.data[6]);

        taskIdx += total_thread;
    }
}

void gen_rays_cuda(
    at::Tensor view_idxs, at::Tensor ray_idxs,
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &rays_o, at::Tensor &rays_d, at::Tensor &up_axis,
    int height, int width)
{
    int batch_size = rays_o.size(0);

    int num_camera = rts.size(0);


    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);



    gen_rays_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (int*)view_idxs.contiguous().data_ptr<int>(),
        (int*)ray_idxs.contiguous().data_ptr<int>(),
        cameras, 
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)up_axis.contiguous().data_ptr<float>(),
        height, width, batch_size);


    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


__global__ 
void gen_image_rays_kernel(
    PinholeCameraManager cameras, 
    int vidx, 
    float3* rays_o, // B x 3
    float3* rays_d, // B x 3 
    float3* up_axis, // B x 3
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < height*width)
    {
        PinholeCamera cam = cameras[vidx];

        float2 uv = make_float2((float)(taskIdx % width) + 0.5f, 
                                (float)(taskIdx / width) + 0.5f);

        float3 x_world = cam.pixel2world(uv, 1.0f);

        float3 origin = cam.camera_center();

        rays_o[taskIdx] = origin;
        rays_d[taskIdx] = x_world - origin;
        up_axis[taskIdx] = make_float3(cam.E.data[4], cam.E.data[5], cam.E.data[6]);

        taskIdx += total_thread;
    }
}


void gen_image_rays_cuda(
    int vidx,
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &rays_o, at::Tensor &rays_d, at::Tensor &up_axis,
    int height, int width)
{
    int num_camera = rts.size(0);


    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);



    gen_image_rays_kernel<<<NUM_BLOCK(height*width), NUM_THREAD>>>(
        cameras, vidx,
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)up_axis.contiguous().data_ptr<float>(),
        height, width);


    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


__global__ 
void get_candidate_neighbor_kernel(
    int* ref_idxs, // B x 1
    float3* rays_o, // B x 3 
    PinholeCameraManager cameras, // captured cameras 
    float* distance, // B x num_camera 
    int num_camera, int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        // PinholeCamera ref_cam = cameras[taskIdx];

        if (ref_idxs[taskIdx] == blockIdx.y)
        {
            taskIdx += total_thread;
            continue;
        }
        PinholeCamera nei_cam = cameras[blockIdx.y];

        // float3 ref_origin = ref_cam.camera_center();
        float3 ref_origin = rays_o[taskIdx];
        float3 nei_origin = nei_cam.camera_center();

        // only consider distance right now 
        distance[taskIdx * num_camera + blockIdx.y] = norm(ref_origin - nei_origin);

        taskIdx += total_thread;
    }
}

void get_candidate_neighbor(
    at::Tensor ref_idxs, at::Tensor rays_o, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor distance)
{
    int batch_size = rays_o.size(0);
    int num_camera = rts.size(0);

    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    get_candidate_neighbor_kernel<<<dim3(NUM_BLOCK(batch_size), num_camera), NUM_THREAD>>>(
        (int*)ref_idxs.contiguous().data_ptr<int>(),
        (float3*)rays_o.contiguous().data_ptr<float>(), cameras,
        (float*)distance.contiguous().data_ptr<float>(),
        num_camera, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}

__device__ 
float point_line_distance(
    float3 point,
    float3 line_origin,
    float3 line_direction)
{
    return norm(cross(normalize(line_direction), point - line_origin));
}


__global__ 
void pixel_level_neighbor_ranking_kernel(
    int* ref_idxs, // B X 1 
    int* candidate_neighbors, // B x K 
    PinholeCameraManager cameras, // captured cameras 
    float3* rays_o, // B x 3 
    float3* rays_d, // B x 3 
    float* z_vals, // B x num_sample x 1  TODO z_vals
    float3* up_axis, // B x 3
    float* score, // B x num_step x K x 4 
    int step, int padding,
    int height, int width,
    int num_step,
    int num_candidate, // K
    int num_sample,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int point_idx = taskIdx % num_sample;

        float3 origin = rays_o[batch_idx];
        float3 direction = rays_d[batch_idx];
        float z_depth = z_vals[taskIdx];
        float3 x_world = origin + z_depth * direction;

        int ref_idx = ref_idxs[batch_idx];
        int nei_idx = candidate_neighbors[batch_idx*num_candidate+blockIdx.y];

        if (ref_idx == nei_idx)
        {
            taskIdx += total_thread;
            continue;
        }

        PinholeCamera cam = cameras[nei_idx];

        float3 x_pixel = cam.world2pixel(x_world);
        if (x_pixel.z <= 0)
        {
            taskIdx += total_thread;
            continue;
        }
        float2 uv = make_float2(x_pixel.x / x_pixel.z, x_pixel.y / x_pixel.z);
        if (uv.x < -padding || uv.y < -padding || uv.x >= width+padding || uv.y >= height+padding)
        {
            taskIdx += total_thread;
            continue;
        }

        float3 camera_center = cam.camera_center();

        float3 up = up_axis[batch_idx];

        float3 view_drection = normalize(camera_center - x_world);
        float3 normal = normalize(origin - x_world);



        // float3 basis_a = normalize(cross(normal, up));
        // float3 basis_b = normalize(cross(normal, basis_a));

        // float coor_a = dot(view_drection, basis_a);
        // float coor_b = dot(view_drection, basis_b); 
        // float coor_c = dot(view_drection, normal);

        // int region = ((signf(coor_b) + 1) + (signf(coor_a) + 1) / 2);

        float cos_sim = dot(view_drection, normal);

        float3 vec = camera_center - origin;


        float3 basis_a = normalize(cross(normal, up));
        float3 basis_b = normalize(cross(normal, basis_a));

        float coor_a = dot(vec, basis_a);
        float coor_b = dot(vec, basis_b); 

        int region = ((signf(coor_b) + 1) + (signf(coor_a) + 1) / 2);

        int step_idx = point_idx / step; 
        atomicAdd(score+(batch_idx*num_step*num_candidate+step_idx*num_candidate+blockIdx.y)*4+region, cos_sim);
        // atomicAdd(score+(batch_idx*num_candidate+blockIdx.y) * 4 + region, coor_c);

        taskIdx += total_thread;
    }
}

void pixel_level_neighbor_ranking(
    at::Tensor ref_idxs,
    at::Tensor candidate_neighbors, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, 
    at::Tensor up_axis,
    at::Tensor score, 
    int step, int padding, 
    int height, int width)
{
    int batch_size = rays_o.size(0);
    int num_sample = z_vals.size(1);
    int num_camera = rts.size(0);
    int num_candidate = candidate_neighbors.size(1);
    int num_step = score.size(1);

    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    pixel_level_neighbor_ranking_kernel<<<dim3(NUM_BLOCK(batch_size*num_sample), num_candidate), NUM_THREAD>>>(
        (int*)ref_idxs.contiguous().data_ptr<int>(),
        (int*)candidate_neighbors.contiguous().data_ptr<int>(),
        cameras,
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float*)z_vals.contiguous().data_ptr<float>(),
        (float3*)up_axis.contiguous().data_ptr<float>(),
        (float*)score.contiguous().data_ptr<float>(),
        step, padding, height, width, num_step,
        num_candidate, num_sample, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


__global__ 
void pixel_level_pick_up_neighbor_kernel(
    int* sorted_idxs, //  B x num_candidate x 4 
    int* candidate_neighbors, // B x num_candidate 
    int* out, // B x num_candidate
    int num_candidate,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        int* cur_sorted_idxs = sorted_idxs + taskIdx * num_candidate * 4;
        int* cur_out = out + taskIdx * num_candidate;
        int* cur_candidate_neighbors = candidate_neighbors + taskIdx * num_candidate;
        int count = 0;
        for (int i=0; i<num_candidate*4 && count < num_candidate; i++)
        {
            if(cur_sorted_idxs[i] != -1)
            {

                // printf("i %d sorted_idx %d vidx %d\n", i, cur_sorted_idxs[i], cur_candidate_neighbors[cur_sorted_idxs[i]]);
                cur_out[count++] = cur_candidate_neighbors[cur_sorted_idxs[i]];

                // if (taskIdx == 149509)
                // {
                //     printf("i %d sorted_idx %d vidx %d out %d count %d\n", 
                //     i, cur_sorted_idxs[i], cur_candidate_neighbors[cur_sorted_idxs[i]], cur_out[count-1], count);
                // }

            }
        }

        taskIdx += total_thread;
    }
}

void pixel_level_pick_up_neighbor(
    at::Tensor sorted_idxs, 
    at::Tensor candidate_neighbors,
    at::Tensor out)
{
    int batch_size = out.size(0);
    int num_candidate = out.size(1);

    pixel_level_pick_up_neighbor_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (int*)sorted_idxs.contiguous().data_ptr<int>(),
        (int*)candidate_neighbors.contiguous().data_ptr<int>(),
        (int*)out.contiguous().data_ptr<int>(),
        num_candidate, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


// __global__ 
// void new_neighbor_score_kernel(
//     int* ref_idxs, // B x 1
//     float3* up_axis,
//     float3* rays_o, // B x 3 
//     float3* rays_d,
//     float3* samples, // B x num_sample x 3 
//     PinholeCameraManager cameras, 
//     float* score, // B x num_sample x num_camera x 4
//     // short* region, // B x num_sample x num_camera x 1
//     int batch_size,
//     int num_sample,
//     int num_camera,
//     int height, int width)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 
//     while(taskIdx < batch_size*num_sample)
//     {
//         int batch_idx = taskIdx / num_sample;
//         int sample_idx = taskIdx % num_sample;

//         int ref_idx = ref_idxs[batch_idx];
//         int neighbor_idx = blockIdx.y;

//         if (ref_idx == neighbor_idx)
//         {
//             taskIdx += total_thread;
//             continue;
//         }

//         float3 origin = rays_o[batch_idx];
//         float3 direction = rays_d[batch_idx];
//         float3 x_world = samples[taskIdx];
        
//         PinholeCamera cam = cameras[neighbor_idx];


//         float3 x_cam = cam.world2cam(x_world);
        
//         float3 x_pixel = cam.cam2pixel(x_cam);

//         // if (batch_idx == 493 && neighbor_idx == 8 && sample_idx == 0)
//         // {
//         //     printf("x_world %f %f %f\nx_cam %f %f %f\nx_pixel %f %f %f\n",
//         //     x_world.x, x_world.y, x_world.z,
//         //     x_cam.x, x_cam.y, x_cam.z,
//         //     x_pixel.x, x_pixel.y, x_pixel.z);
//         // }

//         if (x_pixel.z <= 0)
//         {
//             // if (batch_idx == 493 && neighbor_idx == 8)
//             // {
//             //     printf("x_pixel %f %f %f\n", x_pixel.x, x_pixel.y, x_pixel.z);
//             // }
//             taskIdx += total_thread;
//             continue;
//         }


//         float2 uv = make_float2(x_pixel.x / x_pixel.z, x_pixel.y / x_pixel.z);
//         if (uv.x < 0 || uv.y < 0 || uv.x > width || uv.y > height)
//         {
//             // if (batch_idx == 493 && neighbor_idx == 8)
//             // {
//             //     printf("uv %f %f\n", uv.x, uv.y);
//             // }
//             taskIdx += total_thread;
//             continue;
//         }

//         float3 camera_center = cam.camera_center();

//         float3 up = up_axis[batch_idx];

//         // float3 up = make_float3(0.0f,1.0f,0.0f);

//         // project the view_direction to two-basis
//         float3 view_drection = normalize(camera_center - x_world);

//         // the normal of the plane 
//         float3 normal = normalize(origin - x_world);


//         float3 basis_a = normalize(cross(normal, up));
//         float3 basis_b = normalize(cross(normal, basis_a));

//         float coor_a = dot(view_drection, basis_a);
//         float coor_b = dot(view_drection, basis_b); 
//         float coor_c = dot(view_drection, normal);

//         // score[taskIdx*num_camera + neighbor_idx] = sqrtf(coor_a * coor_a + coor_b * coor_b);
//         int region = ((signf(coor_b) + 1) + (signf(coor_a) + 1) / 2);

//         score[(taskIdx*num_camera+neighbor_idx)*4 + region] = coor_c;

//         // score[taskIdx*num_camera*4 + neighbor_idx*4 + region] = 1.0f - e_ibr(x_world, origin, camera_center);

//         taskIdx += total_thread;
//     }
// }

// void new_neighbor_score_cuda(
//     at::Tensor ref_idxs,
//     at::Tensor up_axis,
//     at::Tensor rays_o, 
//     at::Tensor rays_d,
//     at::Tensor samples, 
//     at::Tensor depths, float dthresh,
//     at::Tensor rts, at::Tensor ks, 
//     at::Tensor &score,
//     int height, int width)
// {
//     int batch_size = samples.size(0);
//     int num_sample = samples.size(1);
//     int num_camera = rts.size(0);


//     PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
//                                 (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);



//     new_neighbor_score_kernel<<<dim3(NUM_BLOCK(batch_size*num_sample), num_camera), NUM_THREAD>>>(
//         (int*)ref_idxs.contiguous().data_ptr<int>(),
//         (float3*)up_axis.contiguous().data_ptr<float>(),
//         (float3*)rays_o.contiguous().data_ptr<float>(),
//         (float3*)rays_d.contiguous().data_ptr<float>(),
//         (float3*)samples.contiguous().data_ptr<float>(), cameras,
//         (float*)score.contiguous().data_ptr<float>(),
//         batch_size, num_sample, num_camera, height, width);


//     AT_CUDA_CHECK(cudaGetLastError());
//     return;

// }


// __global__ 
// void pick_up_neighbor_kernel(
//     int* nei_idx,  // B x num_sample x num_view x 4
//     int* out, // B x num_sample x num_view x 1
//     int batch_size,
//     int num_sample,
//     int num_view)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 
//     while(taskIdx < batch_size*num_sample)
//     {
//         int* cur_nei_idx = nei_idx + taskIdx * num_view * 4;
//         int* cur_out = out + taskIdx * num_view;

//         int count = 0;
//         for (int i=0; i<num_view*4; i++)
//         {
//             if (cur_nei_idx[i] != -1)
//             {
//                 cur_out[count++] = cur_nei_idx[i];
//             }
//         }
//         assert (count <= num_view);


//         taskIdx += total_thread;
//     }
// }

// void pick_up_neighbor(
//     at::Tensor nei_idx, 
//     at::Tensor &out)
// {
//     int batch_size = nei_idx.size(0);
//     int num_sample = nei_idx.size(1);
//     int num_camera = nei_idx.size(2);

//     pick_up_neighbor_kernel<<<NUM_BLOCK(batch_size*num_sample), NUM_THREAD>>>(
//         (int*)nei_idx.contiguous().data_ptr<int>(),
//         (int*)out.contiguous().data_ptr<int>(), batch_size, 
//         num_sample, num_camera);

//     AT_CUDA_CHECK(cudaGetLastError());
//     return;
// }


///////// View selection ///////////////////



__global__ 
void get_candidate_uniform_neighbor_kernel(
    int* ref_idxs, // B x 1
    float* c2ws, // B x 3 x 4
    float3* nei_camera_centers, // num_camera x 3
    float* distance, // B x num_camera x 4 
    int num_camera, int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        if (ref_idxs[taskIdx] == blockIdx.y)
        {
            taskIdx += total_thread;
            continue;
        }

        float* cur_c2w = c2ws + taskIdx * 12;

        float3 ref_origin = make_float3(cur_c2w[3], cur_c2w[7], cur_c2w[11]);
        float3 nei_origin = nei_camera_centers[blockIdx.y];


        float3 r_nei_origin = nei_origin - ref_origin;

        float3 right_axis = make_float3(cur_c2w[0],cur_c2w[4],cur_c2w[8]);
        float3 up_axis = make_float3(cur_c2w[1],cur_c2w[5],cur_c2w[9]);

        float u = dot(r_nei_origin, right_axis);
        float v = dot(r_nei_origin, up_axis);

        int region = ((signf(v) + 1) + (signf(u) + 1) / 2);

        float d = norm(r_nei_origin);

        distance[(taskIdx * num_camera + blockIdx.y) * 4 + region] = d;

        taskIdx += total_thread;
    }
}

void get_candidate_uniform_neighbor(
    at::Tensor ref_idxs,  at::Tensor c2ws,
    at::Tensor nei_camera_centers,
    at::Tensor distance)
{
    int num_camera = nei_camera_centers.size(0);
    int batch_size = c2ws.size(0);

    get_candidate_uniform_neighbor_kernel<<<dim3(NUM_BLOCK(batch_size), num_camera), NUM_THREAD>>>(
        (int*)ref_idxs.contiguous().data_ptr<int>(),
        (float*)c2ws.contiguous().data_ptr<float>(),
        (float3*)nei_camera_centers.contiguous().data_ptr<float>(),
        (float*)distance.contiguous().data_ptr<float>(),
        num_camera, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}   

///////// View selection ///////////////////

__global__ 
void new_neighbor_score_kernel(
    int* ref_idxs, // B x 1
    float3* up_axis, // B x 3
    float3* rays_o, // B x 3 
    float3* rays_d,
    float3* samples, // B x num_sample x 3 
    PinholeCameraManager cameras, 
    int* candidate_neighbors, // B x num_candidate
    float* score, // B x num_sample x num_camera x 4
    int batch_size,
    int num_sample,
    int num_camera,
    int num_candidate,
    int padding, 
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;

        int ref_idx = ref_idxs[batch_idx];
        int neighbor_idx = candidate_neighbors[batch_idx * num_candidate + blockIdx.y];

        if (ref_idx == neighbor_idx)
        {
            taskIdx += total_thread;
            continue;
        }

        float3 origin = rays_o[batch_idx];
        float3 direction = rays_d[batch_idx];
        float3 x_world = samples[taskIdx];
        
        PinholeCamera cam = cameras[neighbor_idx];


        float3 x_cam = cam.world2cam(x_world);
        
        float3 x_pixel = cam.cam2pixel(x_cam);

        if (x_pixel.z <= 0)
        {
            taskIdx += total_thread;
            continue;
        }

        
        float2 uv = make_float2(x_pixel.x / x_pixel.z, x_pixel.y / x_pixel.z);
        // if (uv.x < 0 || uv.y < 0 || uv.x > width || uv.y > height)
        // {
        //     taskIdx += total_thread;
        //     continue;
        // }
        if (uv.x < -padding || uv.y < -padding || uv.x >= width+padding || uv.y >= height+padding)
        {
            taskIdx += total_thread;
            continue;
        }

        float3 camera_center = cam.camera_center();

        float3 up = up_axis[batch_idx];

        // project the view_direction to two-basis
        float3 view_drection = normalize(camera_center - x_world);

        // the normal of the plane 
        float3 normal = normalize(origin - x_world);

        float3 basis_a = normalize(cross(normal, up));
        float3 basis_b = normalize(cross(normal, basis_a));

        float coor_a = dot(view_drection, basis_a);
        float coor_b = dot(view_drection, basis_b); 
        float coor_c = dot(view_drection, normal);

        int region = ((signf(coor_b) + 1) + (signf(coor_a) + 1) / 2);

        score[(taskIdx*num_candidate+blockIdx.y)*4 + region] = coor_c;
        taskIdx += total_thread;
    }
}

void new_neighbor_score_cuda(
    at::Tensor ref_idxs,
    at::Tensor up_axis,
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor samples, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor candidate_neighbors,
    at::Tensor &score, 
    int padding,
    int height, int width)
{
    int batch_size = samples.size(0);
    int num_sample = samples.size(1);
    int num_camera = rts.size(0);
    int num_candidate = candidate_neighbors.size(1);


    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);



    new_neighbor_score_kernel<<<dim3(NUM_BLOCK(batch_size*num_sample), num_candidate), NUM_THREAD>>>(
        (int*)ref_idxs.contiguous().data_ptr<int>(),
        (float3*)up_axis.contiguous().data_ptr<float>(),
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)samples.contiguous().data_ptr<float>(), cameras,
        (int*)candidate_neighbors.contiguous().data_ptr<int>(),
        (float*)score.contiguous().data_ptr<float>(),
        batch_size, num_sample, num_camera, num_candidate, padding, height, width);


    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


__global__ 
void pick_up_neighbor_kernel(
    int* nei_idx,  // B x num_sample x num_view x 4
    int* candidate_neighbors, // B x num_candidate 
    float* nei_scores, // B x num_candidate 
    int* out, // B x num_sample x num_view x 1
    int batch_size,
    int num_sample,
    int num_view)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int* cur_nei_idx = nei_idx + taskIdx * num_view * 4;
        int* cur_out = out + taskIdx * num_view;
        int* cur_candidate_neighbors = candidate_neighbors + batch_idx * num_view;
        float* cur_nei_scores = nei_scores + batch_idx * num_view;

        int count = 0;
        for (int i=0; i<num_view*4 && count < num_view; i++)
        {
            // if (cur_nei_scores[i] > 0)
            // {
                // cur_out[count++] = cur_candidate_neighbors[cur_nei_idx[i]];
            // }
            if (cur_nei_idx[i] != -1)
            {
                cur_out[count++] = cur_candidate_neighbors[cur_nei_idx[i]];
            }
        }

        // backup 
        // for (int i=0; i<num_view*4 && count < num_view; i++)
        // {
        //     if (cur_nei_scores[i] <= 0 && cur_nei_scores[i] >= -1)
        //     {
        //         cur_out[count++] = cur_candidate_neighbors[cur_nei_idx[i]];
        //     }
        // }
        assert (count <= num_view);


        taskIdx += total_thread;
    }
}

void pick_up_neighbor(
    at::Tensor nei_idx, 
    at::Tensor candidate_neighbors,
    at::Tensor nei_scores, 
    at::Tensor &out)
{
    int batch_size = nei_idx.size(0);
    int num_sample = nei_idx.size(1);
    int num_camera = nei_idx.size(2);

    pick_up_neighbor_kernel<<<NUM_BLOCK(batch_size*num_sample), NUM_THREAD>>>(
        (int*)nei_idx.contiguous().data_ptr<int>(),
        (int*)candidate_neighbors.contiguous().data_ptr<int>(),
        (float*)nei_scores.contiguous().data_ptr<float>(),
        (int*)out.contiguous().data_ptr<int>(), batch_size, 
        num_sample, num_camera);

    AT_CUDA_CHECK(cudaGetLastError());
    return;


}

__global__ 
void neighbor_score_kernel(
    float3* rays_o, // B x 3 
    float3* samples, // B x num_sample x 3 
    float* depths, // num_camera x H x W x 1
    float dthresh, 
    PinholeCameraManager cameras, 
    float* score, // B x num_sample x num_camera x 1
    int batch_size,
    int num_sample,
    int num_camera,
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;

        float3 origin = rays_o[batch_idx];
        float3 x_world = samples[taskIdx];
        int neighbor_idx = blockIdx.y;
        PinholeCamera cam = cameras[neighbor_idx];

        // float* cur_depth = depths + neighbor_idx * height * width;

        float3 x_cam = cam.world2cam(x_world);
        float3 x_pixel = cam.cam2pixel(x_cam);

        if (x_pixel.z <= 0)
        {
            taskIdx += total_thread;
            continue;
        }

        // [FIXME] here
        float2 uv = make_float2(x_pixel.x / x_pixel.z, x_pixel.y / x_pixel.z);
        if (uv.x <= 0 || uv.y <=0 || uv.x >= width-1 || uv.y >= height-1)
        {
            taskIdx += total_thread;
            continue;
        }

        // float nei_depth = cur_depth[ int(uv.y) * width + (int)uv.x ];


        // float vis_score = 1.0f;
        // float depth_diff = x_cam.z - nei_depth;

        // if (depth_diff > 0)
        // {
        //     vis_score = expf(-1.0f * depth_diff * depth_diff * dthresh);
        // }

        score[taskIdx*num_camera + neighbor_idx] = 1.0f - e_ibr(x_world, origin, cam.camera_center());
        // float3 camera_center = cam.camera_center();
        // float3 ray_neighbor = x_world - camera_center;
        // float S = norm(cross(ray_neighbor, origin - camera_center));
        // float dis = S / norm(ray_neighbor);
        // score[taskIdx*num_camera + neighbor_idx] = expf(-1.0 * dis * dis);
        taskIdx += total_thread;
    }
}



void neighbor_score_cuda(
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor samples, 
    at::Tensor depths, float dthresh,
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &score,
    int height, int width)
{
    int batch_size = samples.size(0);
    int num_sample = samples.size(1);
    int num_camera = rts.size(0);


    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    
    neighbor_score_kernel<<<dim3(NUM_BLOCK(batch_size*num_sample), num_camera), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)samples.contiguous().data_ptr<float>(), 
        (float*)depths.contiguous().data_ptr<float>(), 
        dthresh, cameras,
        (float*)score.contiguous().data_ptr<float>(),
        batch_size, num_sample, num_camera, height, width);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}




// __global__ 
// void inference_per_view_kernel(
//     float3* rays_o, // B x 3
//     float3* samples, // B x num_sample x 3
//     float* src_views, // num_view x H x W x (3 * num_gaussian)
//     float* score, // B x num_sample x num_view x 1
//     PinholeCameraManager cameras, // num_view
//     int num_sample, int num_view, 
//     int num_gaussian, int num_neighbor,
//     int height, int width, int batch_size)
// {
//     // total task is B x num_sample x num_view

//     // Each thread is responsible for infering a neighbor for a single sample point.
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 

//     int view_channel = num_gaussian * 3;

//     while(taskIdx < batch_size*num_sample*num_view)
//     {

//         int bidx = taskIdx / (num_sample*num_view);
//         int pidx = taskIdx % (num_sample*num_view) / num_view;
//         int vidx = taskIdx % (num_sample*num_view) % num_view;
        
//         // current sample point 
//         float3 x_world = samples[bidx * num_sample + pidx];
//         // current neighbor camera 
//         PinholeCamera cam = cameras[vidx];
//         // current neighbor view 
//         float* neighbor_view = src_views + vidx * height * width * view_channel;
        

//         // project the sample to neighbor camera space
//         float3 x_cam = cam.world2cam(x_world);



//         taskIdx += total_thread;
//     }
// } 




__global__ 
void project_neighbor_backward_kernel(
    float3* grad_pts, // B x 3 
    int* nei_idxs, // B x num_neighbor
    PinholeCameraManager cameras, // N   world 2 camera 
    float3* grid_in, // B x num_neighbor x 3
    bool* mask, // B x num_neighbor x 1
    int num_neighbor,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int nei_idx = nei_idxs[taskIdx *  num_neighbor + blockIdx.y];
        PinholeCamera cam = cameras[nei_idx];

        if (!mask[taskIdx *  num_neighbor + blockIdx.y])
        {
            taskIdx += total_thread;
            continue;
        }

        float3 dout_dx = grid_in[taskIdx *  num_neighbor + blockIdx.y] * cam.K.proj(make_float3(cam.E.data[0], cam.E.data[4], cam.E.data[8]));
        float3 dout_dy = grid_in[taskIdx *  num_neighbor + blockIdx.y] * cam.K.proj(make_float3(cam.E.data[1], cam.E.data[5], cam.E.data[9]));
        float3 dout_dz = grid_in[taskIdx *  num_neighbor + blockIdx.y] * cam.K.proj(make_float3(cam.E.data[2], cam.E.data[6], cam.E.data[10]));

        atomicAdd(&grad_pts[taskIdx].x, dout_dx.x+dout_dx.y+dout_dx.z);
        atomicAdd(&grad_pts[taskIdx].y, dout_dy.x+dout_dy.y+dout_dy.z);
        atomicAdd(&grad_pts[taskIdx].z, dout_dz.x+dout_dz.y+dout_dz.z);
        taskIdx += total_thread;
    }
}

void project_neighbor_backward_cuda(
    at::Tensor &grad_pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor grad_in, at::Tensor mask)
{
    int batch_size = grad_pts.size(0);
    int num_neighbor = nei_idxs.size(1);
    int num_camera = rts.size(0);

    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    
    project_neighbor_backward_kernel<<<dim3(NUM_BLOCK(batch_size), num_neighbor), NUM_THREAD>>>(
        (float3*)grad_pts.contiguous().data_ptr<float>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(),
        cameras,
        (float3*)grad_in.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(), num_neighbor, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


__global__ 
void get_neighbor_color_forward_kernel(
    float3* rays_o, // B x 3 
    float3* pts, // B x 3
    int* nei_idxs, // B x num_neighbor 
    PinholeCameraManager cameras, // N   world 2 camera 
    uint8_t* images, // N x H x W x 3 
    float* block_center, float* block_size, // 3 
    float* ray_distance, //B x num_neighbor x 1 
    float2* nei_bound, // B x num_neighbor x 2
    float3* warped_uvs, // B x num_neighbor x 3
    float3* warped_color, // B x num_neighbor x 3 
    float* blend_weight, // B x num_neighbor x 1
    int height, int width, 
    int num_neighbor, int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 x_world = pts[taskIdx];
        float3 origin = rays_o[taskIdx];

        int nei_idx = nei_idxs[taskIdx *  num_neighbor + blockIdx.y];

        if (nei_idx == -1)
        {
            blend_weight[taskIdx *  num_neighbor + blockIdx.y] = 0.0f;
            taskIdx += total_thread;
            continue;
        }
        PinholeCamera cam = cameras[nei_idx];

        float3 x_cam = cam.world2cam(x_world);
        float3 pixel = cam.cam2pixel(x_cam);
        float3 camera_center = cam.camera_center();
        float3 nei_ray_d = normalize(x_world - camera_center);

        if (pixel.z <= 0)
        {
            blend_weight[taskIdx *  num_neighbor + blockIdx.y] = 0.0f;
            taskIdx += total_thread;
            continue;
        }
        float2 uv = make_float2(pixel.x / pixel.z, pixel.y / pixel.z);
        uint8_t* cur_image = images + nei_idx * height * width * 3;

        bool valid;
        float3 color = fetch_color_forward(cur_image, uv, height, width, valid);
        if (!valid)
        {
            blend_weight[taskIdx *  num_neighbor + blockIdx.y] = 0.0f;
            taskIdx += total_thread;
            continue;
        }
        
        blend_weight[taskIdx * num_neighbor + blockIdx.y] = 1.0f - e_ibr(x_world, origin, camera_center);
        warped_color[taskIdx * num_neighbor + blockIdx.y] = color;
        warped_uvs[taskIdx * num_neighbor + blockIdx.y] = pixel;
        nei_bound[taskIdx * num_neighbor + blockIdx.y] = RayAABBIntersection(camera_center, nei_ray_d, 
                                        make_float3(block_center[0],block_center[1],block_center[2]),
                                        make_float3(block_size[0],block_size[1],block_size[2]) / 2.0f);
        ray_distance[taskIdx * num_neighbor + blockIdx.y] = norm(x_cam - camera_center);

        taskIdx += total_thread;
    }
}

void get_neighbor_color_forward_cuda(
    at::Tensor rays_o,
    at::Tensor pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor images, at::Tensor block_center,
    at::Tensor block_size, at::Tensor &ray_distance,
    at::Tensor &nei_bound, at::Tensor &warped_uvs,
    at::Tensor &warped_color, 
    at::Tensor &blend_weight)
{
    int batch_size = pts.size(0);
    int num_neighbor = nei_idxs.size(1);
    int num_camera = rts.size(0);
    int height = images.size(1);
    int width = images.size(2);

    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    
    get_neighbor_color_forward_kernel<<<dim3(NUM_BLOCK(batch_size), num_neighbor), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)pts.contiguous().data_ptr<float>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(), cameras, 
        (uint8_t*)images.contiguous().data_ptr<uint8_t>(),
        (float*)block_center.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(),
        (float*)ray_distance.contiguous().data_ptr<float>(),
        (float2*)nei_bound.contiguous().data_ptr<float>(),
        (float3*)warped_uvs.contiguous().data_ptr<float>(),
        (float3*)warped_color.contiguous().data_ptr<float>(),
        (float*)blend_weight.contiguous().data_ptr<float>(),
        height, width, num_neighbor, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}

__global__ 
void get_neighbor_color_backward_kernel(
    float3* grad_pts, // B x 3 
    float3* pts, // B x 3
    int* nei_idxs, // B x num_neighbor 
    PinholeCameraManager cameras, // N   world 2 camera 
    uint8_t* images, // N x H x W x 3 
    float3* grad_in, // B x num_neighbor x 3 
    int height, int width, 
    int num_neighbor, int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 x_world = pts[taskIdx];
        int nei_idx = nei_idxs[taskIdx *  num_neighbor + blockIdx.y];

        if (nei_idx == -1)
        {
            taskIdx += total_thread;
            continue;
        }

        PinholeCamera cam = cameras[nei_idx];


        float3 pixel = cam.world2pixel(x_world);

        if (pixel.z <= 0)
        {
            taskIdx += total_thread;
            continue;
        }
        float2 uv = make_float2(pixel.x / pixel.z, pixel.y / pixel.z);
        uint8_t* cur_image = images + nei_idx * height * width * 3;
        float2 dL_duv = fetch_color_backward(cur_image, uv, height, width, 
                            grad_in[taskIdx *  num_neighbor + blockIdx.y]);

        float3 dL_dpixel = make_float3(dL_duv.x * 1.0f / pixel.z, 
                                       dL_duv.y * 1.0f / pixel.z,
                                      -1.0f * dL_duv.x * pixel.x * 1.0f / (pixel.z * pixel.z) 
                                      -1.0f * dL_duv.y * pixel.y * 1.0f / (pixel.z * pixel.z));

        float3 dL_dx = dL_dpixel * cam.K.proj(make_float3(cam.E.data[0], cam.E.data[4], cam.E.data[8]));
        float3 dL_dy = dL_dpixel * cam.K.proj(make_float3(cam.E.data[1], cam.E.data[5], cam.E.data[9]));
        float3 dL_dz = dL_dpixel * cam.K.proj(make_float3(cam.E.data[2], cam.E.data[6], cam.E.data[10]));


        atomicAdd(&grad_pts[taskIdx].x, dL_dx.x+dL_dx.y+dL_dx.z);
        atomicAdd(&grad_pts[taskIdx].y, dL_dy.x+dL_dy.y+dL_dy.z);
        atomicAdd(&grad_pts[taskIdx].z, dL_dz.x+dL_dz.y+dL_dz.z);

        taskIdx += total_thread;
    }
}

void get_neighbor_color_backward_cuda(
    at::Tensor &grad_pts, 
    at::Tensor pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor images, 
    at::Tensor grad_in)
{
    int batch_size = pts.size(0);
    int num_neighbor = nei_idxs.size(1);
    int num_camera = rts.size(0);
    int height = images.size(1);
    int width = images.size(2);

    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    
    get_neighbor_color_backward_kernel<<<dim3(NUM_BLOCK(batch_size), num_neighbor), NUM_THREAD>>>(
        (float3*)grad_pts.contiguous().data_ptr<float>(),
        (float3*)pts.contiguous().data_ptr<float>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(),
        cameras, 
        (uint8_t*)images.contiguous().data_ptr<uint8_t>(),
        (float3*)grad_in.contiguous().data_ptr<float>(),
        height, width, num_neighbor, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}

__global__ 
void project_neighbor_forward_kernel(
    float3* pts, // B x 3 
    int* nei_idxs, // B x num_neighbor
    PinholeCameraManager cameras, // N   world 2 camera 
    float3* nei_rays_o, // B x num_neighbor x 3
    float3* nei_rays_d, // B x num_neighbor x 3
    float3* proj_uvs, // B x num_neighbor x 3  (uv + depth)
    bool* mask, // B x num_neighbor x 1
    int num_neighbor,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 x_world = pts[taskIdx];
        int nei_idx = nei_idxs[taskIdx *  num_neighbor + blockIdx.y];
        PinholeCamera cam = cameras[nei_idx];

        float3 x_cam = cam.world2cam(x_world);

        if (x_cam.z <= 0)
        {
            mask[taskIdx *  num_neighbor + blockIdx.y] = false;
            taskIdx += total_thread;
            continue;
        }

        float3 camera_center = cam.camera_center();

        float3 x_image_plane = x_cam / x_cam.z;

        proj_uvs[taskIdx * num_neighbor + blockIdx.y] = cam.world2pixel(x_world);

        nei_rays_o[taskIdx * num_neighbor + blockIdx.y] = camera_center;
        nei_rays_d[taskIdx * num_neighbor + blockIdx.y] = cam.cam2world(x_image_plane);

        taskIdx += total_thread;
    }
}


void project_neighbor_forward_cuda(
    at::Tensor pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &nei_rays_o, at::Tensor &nei_rays_d,
    at::Tensor &proj_uvs, at::Tensor &mask)
{
    int batch_size = pts.size(0);
    int num_neighbor = nei_idxs.size(1);
    int num_camera = rts.size(0);

    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    
    project_neighbor_forward_kernel<<<dim3(NUM_BLOCK(batch_size), num_neighbor), NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(),
        cameras, (float3*)nei_rays_o.contiguous().data_ptr<float>(),
        (float3*)nei_rays_d.contiguous().data_ptr<float>(),
        (float3*)proj_uvs.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(), num_neighbor, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


__device__ inline
float2 nearest_valid_neighbor_search(
    bool* cur_valid_mask, // H x W x 1
    int height, int width,
    int x, int y, int max_range)
{
    for (int i=0; i<max_range; i++)
    {
        if (x - i >= 0)
        {
            int idx = y * width + x - i;
            if (cur_valid_mask[idx])
            {
                return make_float2( x - i, y );
            }
        }

        if (x + i < width)
        {
            int idx = y * width + x + i;
            if (cur_valid_mask[idx])
            {
                return make_float2( x + i, y );
            }
        }

        if (y - i >= 0)
        {
            int idx = (y - i) * width + x;
            if (cur_valid_mask[idx])
            {
                return make_float2( x, y - i);
            }
        }

        if (y + i < height)
        {
            int idx = (y + i) * width + x;
            if (cur_valid_mask[idx])
            {
                return make_float2( x, y + i);
            }
        }
    }
    return make_float2(-1.0f, -1.0f);
}


__global__ 
void warping_samples_kernel(
    float3* rays_o, // B x 3 
    float3* ref_up_axis,
    float3* samples, // B x num_sample x 3 
    PinholeCameraManager cameras, // N   world 2 camera 
    uint8_t* images, // N x H x W x 3 
    bool* valid_mask, // N x H x W x 1
    // float* c2ws, // N x 3 x 4
    // float* ks, // N x 3 x 3 
    int* nei_idx, // B x num_sample x num_neighbor x 1
    float3* warped_samples, // B x num_sample x num_neighbor x 3 
    float* ray_distance, // B x num_sample x num_neighbor x 1
    float* block_center, float* block_size, 
    float2* nei_bound, // B x num_sample x num_neighbor x 2 
    float3* nei_dir, // B x num_sample x num_neighbor x 3
    float3* warped_uvs, // B x num_sample x num_neighbor x 3 
    float3* warped_colors, // B x num_sample x num_neighbor x 3 
    float3* warped_colors_blur, // // B x num_sample x num_neighbor x 3 
    bool* warped_mask, // B x num_sample x num_neighbor x 1
    bool* warped_valid_mask, // B x num_sample x num_neighbor x 1
    float* blend_weight, // B x num_sample x num_neighbor x 1
    float sigma, float max_dis,
    int num_sample, 
    int num_neighbor,
    int batch_size, 
    bool inference, 
    int padding, 
    float near, float far,
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;

        float3 origin = rays_o[batch_idx];
        float3 x_world = samples[taskIdx];

        int current_index = taskIdx * num_neighbor + blockIdx.y;
        int neighbor_idx = nei_idx[current_index];

        if (neighbor_idx == -1)
        {
            warped_mask[current_index] = false;
            taskIdx += total_thread;
            continue;
        }
        
        PinholeCamera cam = cameras[neighbor_idx];
        
        warped_mask[current_index] = true;

        float3 x_cam = cam.world2cam(x_world);

        float3 x_pixel = cam.cam2pixel(x_cam);

        warped_samples[current_index] = x_cam;

        float3 camera_center = cam.camera_center();
        ray_distance[current_index] = norm(x_world - camera_center);

        float3 nei_ray_d = normalize(x_world - camera_center);

        float3 target_dir = normalize(x_world - origin);

        
        // float3 forward_axis = normalize(x_world - origin);
        // float3 right_axis = cross(ref_up_axis[batch_idx], forward_axis);
        // float3 up_axis = cross(forward_axis, right_axis);

        // nei_dir[current_index] = make_float3(dot(nei_ray_d, right_axis),
        //                                      dot(nei_ray_d, up_axis),
        //                                      dot(nei_ray_d, forward_axis));

        nei_dir[current_index] = target_dir - nei_ray_d;

        // nei_dir[current_index] = nei_ray_d;

        // constract coor at neighbor view
        // float3 forward_axis = normalize(x_world - camera_center);
        // float3 right_axis = cross(make_float3(cam.E.data[4], cam.E.data[5], cam.E.data[6]), forward_axis);
        // float3 up_axis = cross(forward_axis, right_axis);

        // nei_dir[current_index] = make_float3(dot(target_dir, right_axis),
        //                                      dot(target_dir, up_axis),
        //                                      dot(target_dir, forward_axis));




        if (x_pixel.z <= 0)
        {
            warped_mask[current_index] = false;
            taskIdx += total_thread;
            continue;
        }
        

        float2 uv = make_float2(x_pixel.x / x_pixel.z, x_pixel.y / x_pixel.z);


        float2 bound = RayAABBIntersection(camera_center, nei_ray_d, 
                                        make_float3(block_center[0],block_center[1],block_center[2]),
                                        make_float3(block_size[0],block_size[1],block_size[2]) / 2.0f);

        float scale = norm(cam.pixel2imageplane(make_float2((int)uv.x, (int)uv.y)));
        bound.x = fmaxf(bound.x, near * scale);
        bound.y = fminf(bound.y, far * scale);

        nei_bound[current_index] = bound;


        if (uv.x < -padding || uv.x >= width+padding || uv.y < -padding || uv.y >= height+padding)
        {
            warped_mask[current_index] = false;
        }else{
            warped_mask[current_index] = true;
        }

        // if (inference == true)
        // {
        //     uv.x = clamp(uv.x, 0.0001f, width-0.0001f);
        //     uv.y = clamp(uv.y, 0.0001f, height-0.0001f);
        // }

        warped_uvs[current_index] = make_float3(uv.x, uv.y, x_pixel.z);

        bool valid = true;
        float3 color = fetch_color(images + neighbor_idx * height * width * 3,
                                   uv, height, width, valid);
        // warped_mask[current_index] = valid;
        // float3 blur_color = gaussian_interpolate(images + neighbor_idx * height * width * 3,
        //                                         uv, height, width, sigma, max_dis);

        // if (batch_idx == 698 && sample_idx == num_sample - 1)
        // {
        //     printf("neiIdx %d uv %f %f color %f %f %f world space %f %f %f\n", neighbor_idx, 
        //     uv.x, uv.y, color.x, color.y, color.z, x_world.x, x_world.y, x_world.z);
        // }
        // if (!valid)
        // {
        //     taskIdx += total_thread;
        //     continue;
        // }
        // bool* cur_valid_mask = valid_mask + neighbor_idx * height * width * 1;
        // warped_valid_mask[current_index] = cur_valid_mask[ (int)uv.y * width + (int)uv.x ];
        
        // if (cur_valid_mask[ (int)uv.y * width + (int)uv.x ] == false && inference)
        // {
        //     // 如果是inference, 在这里做一个uv跳转
        //     float2 new_uv = nearest_valid_neighbor_search(cur_valid_mask, height, width, (int)uv.x, (int)uv.y, 100);
        //     warped_uvs[current_index] = make_float3(new_uv.x, new_uv.y, x_pixel.z);
        // }
        // warped_mask[current_index] = cur_valid_mask[ (int)uv.y * width + (int)uv.x ];
        // warped_mask[current_index] = true;
        
        // float S = norm(cross(nei_ray_d, origin - camera_center));
        // float dis = S / norm(ray_neighbor);
        // blend_weight[current_index] = expf(-1.0 * dis * dis);
        // dot(normalize(x_world - cam.camera_center()), normalize(x_world - origin));
        // blend_weight[current_index] = 1.0f - e_ibr(x_world, origin, cam.camera_center());

        blend_weight[current_index] = dot(normalize(x_world - cam.camera_center()), normalize(x_world - origin));
        // blend_weight[current_index] = angle_weight(x_world, origin, cam.camera_center());

        warped_colors[current_index] = color;
        // warped_colors_blur[current_index] = blur_color;


        taskIdx += total_thread;
    }
}


void warping_samples_cuda(
    at::Tensor rays_o, 
    at::Tensor up_axis,
    at::Tensor samples, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor images, at::Tensor valid_mask,
    at::Tensor nei_idxs,
    at::Tensor &warped_samples, 
    at::Tensor &ray_distance,
    at::Tensor &nei_bound,
    at::Tensor block_center, at::Tensor block_size, 
    at::Tensor &nei_dir,
    at::Tensor &warped_uvs,
    at::Tensor &warped_colors,
    at::Tensor &warped_colors_blur,
    at::Tensor &warped_mask, at::Tensor &warped_valid_mask,
    at::Tensor &blend_weight, float sigma, float max_dis, 
    float near, float far, 
    bool inference, int padding)
{
    int batch_size = samples.size(0);
    int num_sample = samples.size(1);
    int num_neighbor = nei_idxs.size(2);
    int height = images.size(1);
    int width = images.size(2);
    int num_camera = images.size(0);


    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    
    warping_samples_kernel<<<dim3(NUM_BLOCK(batch_size*num_sample), num_neighbor), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)up_axis.contiguous().data_ptr<float>(),
        (float3*)samples.contiguous().data_ptr<float>(), cameras,
        (uint8_t*)images.contiguous().data_ptr<uint8_t>(),
        (bool*)valid_mask.contiguous().data_ptr<bool>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(), 
        (float3*)warped_samples.contiguous().data_ptr<float>(),
        (float*)ray_distance.contiguous().data_ptr<float>(),
        (float*)block_center.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(), 
        (float2*)nei_bound.contiguous().data_ptr<float>(),
        (float3*)nei_dir.contiguous().data_ptr<float>(),
        (float3*)warped_uvs.contiguous().data_ptr<float>(),
        (float3*)warped_colors.contiguous().data_ptr<float>(),
        (float3*)warped_colors_blur.contiguous().data_ptr<float>(),
        (bool*)warped_mask.contiguous().data_ptr<bool>(),
        (bool*)warped_valid_mask.contiguous().data_ptr<bool>(),
        (float*)blend_weight.contiguous().data_ptr<float>(),
        sigma, max_dis,
        num_sample, num_neighbor, batch_size, inference, padding, 
        near, far, height, width);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}

__global__ 
void gaussian_sample_kernel(
    uint8_t* images, // N x H x W x 3 
    int* view_idx, // B x 1
    float2* uvs, // B x 2
    float3* out_color,
    float sigma, float max_dis,
    int height, int width, int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 


    while(taskIdx < batch_size)
    {
        int idx = view_idx[taskIdx];
        float2 uv = uvs[taskIdx];

        uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
        uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);


        float3 blur_color = gaussian_interpolate(images + idx * height * width * 3,
                                                uv, height, width, sigma, max_dis);
        
        out_color[taskIdx] = blur_color;

        taskIdx += total_thread;
    }
}


void gaussian_sample(
    at::Tensor images,
    at::Tensor view_idx, 
    at::Tensor uvs, at::Tensor &out_color,
    float sigma, float max_dis)
{
    int batch_size = uvs.size(0);
    int height = images.size(1);
    int width = images.size(2);

    gaussian_sample_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (uint8_t*)images.contiguous().data_ptr<uint8_t>(),
        (int*)view_idx.contiguous().data_ptr<int>(), 
        (float2*)uvs.contiguous().data_ptr<float>(),
        (float3*)out_color.contiguous().data_ptr<float>(),
        sigma, max_dis, height, width, batch_size);
    
    AT_CUDA_CHECK(cudaGetLastError());
    return;

}

// __global__ 
// void find_neighbor_kernel(
//     float3* samples,

// )
// {

// }
// __device__ 
// void get_next_region(float3 p, float3 direction,
//                      float3 &new_orgin, float3 &new_direction, 
//                      float3 &region_center, 
//                      float3 &region_size)
// {
//     // outgoing point p 

//     /*
//     |x| = 1; |y| = 1; |z| = 1
//     +x -x +y -y +z -z 
//     */
//     // flag for rotattion 
//     static constexpr float R[6][9] = {{1,0,0,0,0,-1,0,1,0}, // Rx anti-clockwise
//                                       {1,0,0,0,0,1,0,-1,0}, // Rx clockwise
//                                       {0,0,1,0,1,0,-1,0,0}, // Ry anti-clockwise
//                                       {0,0,-1,0,1,0,1,0,0}, // Ry clockwise
//                                       {0,-1,0,1,0,0,0,0,1}, // Rz anti-clockwise
//                                       {0,1,0,-1,0,0,0,0,1}};// Rz clockwise
// /*
//     static constexpr uint8_t flags[24] = {
//        2, // x = 1, |y| <= 1, z >= 1
//        3, // x = 1, |y| <= 1, z <= -1
//        5, // x = 1, |z| <= 1, y >= 1
//        4, // x = 1, |z| <= 1, y <= -1
//        3, // x = -1, |y| <= 1, z >= 1
//        2, // x = -1, |y| <= 1, z <= -1
//        4, // x = -1, |z| <= 1, y >= 1
//        5, // x = -1, |z| <= 1, y <= -1
//        1, // y = 1, |x| <= 1, z >= 1
//        0, // y = 1, |x| <= 1, z <= -1
//        4, // y = 1, |z| <= 1, x >= 1
//        5, // y = 1, |z| <= 1, x <= -1
//        0,1,5,4,
//        0, // z = 1. |x| <= 1, y >= 1
//        1, // z = 1. |x| <= 1, y <= -1
//        3, // z = 1, |y| <= 1, x >= 1
//        2, // z = 1, |y| <= 1, x <= -1
//        1,0,3,2};
    
//     // 找 abs_p 中值最小的维度， 绕着那个axis旋转
//     // 三个维度

//     || <= 1   决定旋转轴，但是不确定  clockwise or anti-clockwise
//     || >= 1 and || = 1 同号， anti-clockwise, 0 异号 clockwise  1
// */ 

//     float3 abs_p = fabs(p);
//     // int offset = (int)(abs_p.x < 1) * 0 + (int)(abs_p.y < 1) * 2 + (int)(abs_p.z < 1) * 4;
    
//     int index = 0;
//     if (abs_p.x < 1)
//         index = (int)(p.y > 0) ^ (int)(p.z > 0) + 0;
//     else if (abs_p.y < 1)
//         index = (int)(p.x > 0) ^ (int)(p.z > 0) + 2;
//     else 
//         index = (int)(p.x > 0) ^ (int)(p.y > 0) + 4;
     
//     // #pragma unroll 
//     // for (int i=0; i<9; i++)
//     //     rotation[i] = R[index][i];
//     float rotation[9] = R[index];


// /*
// x = 1, region center = (1.5, 0, 0)   region size = (1, 2, 2)
// x = -1, region center = (-1.5, 0, 0) region size = (1, 2, 2)
// */
//     if (abs_p.x == 1){
//         region_center = p.x * make_float3(1.5f,0,0);
//         region_size = make_float3(1,2,2);
//     }else if (abs_p.y == 1){
//         region_center = p.y * make_float3(0,1.5f,0);
//         region_size = make_float3(2,1,2);
//     }else{
//         region_center = p.z * make_float3(0,0,1.5f);
//         region_size = make_float3(2,2,1);
//     }

//     // new direction 
//     new_direction.x = rotation[0] * direction.x + rotation[1] * direction.y + rotation[2] * direction.z;
//     new_direction.y = rotation[3] * direction.x + rotation[4] * direction.y + rotation[5] * direction.z;
//     new_direction.z = rotation[6] * direction.x + rotation[7] * direction.y + rotation[8] * direction.z;

//     // new point 
//     p = signf(p) * (abs_p - 1.0f); 
//     new_orgin.x = rotation[0] * p.x + rotation[1] * p.y + rotation[2] * p.z;
//     new_orgin.y = rotation[3] * p.x + rotation[4] * p.y + rotation[5] * p.z;
//     new_orgin.z = rotation[6] * p.x + rotation[7] * p.y + rotation[8] * p.z;

// }

// __device__ 
// void sample_points_single_ray_contract(
//     float3 origin,
//     float3 direction,
//     int num_sample,
//     float3* points,
//     bool* occupied_gird,
//     int3 resolution)
// {

//     // total contract space
//     static constexpr float3 block_center = make_float3(0,0,0);
//     static constexpr float3 block_size = make_float3(4,4,4);
//     static constexpr float3 block_corner = block_center - block_size / 2.0f;

//     // dda 
//     float3 grid_size = block_size / make_float3(resolution);
//     DDASatateScene_v2 dda;

//     // origin is already in contract space 
//     float3 abs_origin = fabs(origin);
//     float3 outside_face = make_float3((float)(abs_origin.x==1), (float)(abs_origin.y==1), (float)(abs_origin.z==1));
//     float3 region_center = outside_face * 1.5f;
//     float3 region_size = 2.0f - outside_face;
//     // bound with first region, not sure if we have second 
//     float2 region_bound = RayAABBIntersection(origin, direction, region_center - region_size/2.0f, region_size/2.0f);

//     // new origin 
//     float3 outgoing_pts = origin + region_bound.y * direction;
//     float3 origin2, direction2, region_center2, region_size2;
//     get_next_region(outgoing_pts, direction, origin2, direction2, region_center2, region_size2);
//     float2 region_bound2 = RayAABBIntersection(origin2, direction2, region_center2 - region_size2/2.0f, region_size2/2.0f);
    
//     dda.init(origin-block_corner, direction, region_bound, resolution, grid_size);
//     float total_length = 0.0f; int count = 0;
//     while(!dda.terminate() && dda.t.x < region_bound.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 total_length += len; count++;
//             }
//         }
//         dda.step();
//     }

//     dda.init(origin2-block_corner, direction2, region_bound2, resolution, grid_size);
//     while(!dda.terminate() && dda.t.x < region_bound2.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 total_length += len; count++;
//             }
//         }
//         dda.step();
//     }

//     if (count == 0) return;
//     dda.init(origin-block_corner, direction, region_bound, resolution, grid_size);
//     int left_sample = num_sample;
//     int sample_count = 0;
//     while(!dda.terminate() && dda.t.x < region_bound.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
//                 if (sample_count == count - 1) 
//                 {
//                     num = left_sample;
//                 }
//                 uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);
//                 left_sample = left_sample-num;
//                 sample_count++;
//             }
//         }

//         dda.step();
//     }
//     dda.init(origin2-block_corner, direction2, region_bound2, resolution, grid_size);
//     while(!dda.terminate() && dda.t.x < region_bound2.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
//                 if (sample_count == count - 1) 
//                 {
//                     num = left_sample;
//                 }
//                 uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);
//                 left_sample = left_sample-num;
//                 sample_count++;
//             }
//         }

//         dda.step();
//     }
// }


__device__ 
void sample_points_single_ray(
    float3 origin,
    float3 direction,
    int num_sample,
    float* z_vals, 
    float3 block_corner, 
    float3 block_size,
    bool* occupied_gird,
    int3 resolution)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    if (bound.x == -1) return;

    // int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;

    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        // uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;

            if (len > 0)
            {
                total_length += len;
                count++;
            }
        }

        dda.step();
    }

    if (count == 0) return;

    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    int left_sample = num_sample;
    int sample_count = 0;
    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        // uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;
            if (len > 0)
            {
                int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
                if (sample_count == count - 1) 
                {
                    num = left_sample;
                }

                uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);

                left_sample = left_sample-num;
                sample_count++;
            }
        }

        dda.step();
    }
}

__global__ 
void sample_points_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample, 
    float* z_vals,
    float3* block_corner,
    float3* block_size,
    bool* occupied_gird,
    int rx, int ry, int rz,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {

        sample_points_single_ray(rays_o[taskIdx], rays_d[taskIdx], num_sample, z_vals+taskIdx*num_sample,
                                block_corner[0], block_size[0], occupied_gird, make_int3(rx,ry,rz));

        taskIdx += total_thread;
    }
}




at::Tensor sample_points_contract(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird)
{
    int batch_size = rays_o.size(0);
    int rx = occupied_gird.size(0);
    int ry = occupied_gird.size(1);
    int rz = occupied_gird.size(2);
    int num_sample = z_vals.size(1);

    sample_points_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        num_sample, z_vals.contiguous().data_ptr<float>(),
        (float3*)block_corner.contiguous().data_ptr<float>(),
        (float3*)block_size.contiguous().data_ptr<float>(),
        (bool*)occupied_gird.contiguous().data_ptr<bool>(),
        rx, ry, rz,
        batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return;
}



// __device__ inline 
// void efficient_sampling_along_ray(
//     float3 origin,
//     float3 direction,
//     float near, float far,
//     int num_sample, 
//     float* t_vals, // num_sample
//     float* weight, // num_sample
//     float thresh)
// {

//     float current_near = -1.0f;
//     float current_far = -1.0f;
//     float total_length = 0.0f;
//     int count = 0;
//     for (int i=0; i<num_sample; i++)
//     {
//         if (weight[i] >= thresh)
//         {
//             float s = (1.0f / t_vals[i] - 1.0f / near) / (1.0f / far - 1.0f / near);

//             if (current_near == -1.0f)
//             {
//                 current_near = s;
//             }else{
//                 current_far = s;
//             }

//         }else{
//             if (current_near == -1.0f)
//             {
//                 // skip
//             }else{
//                 float interval = current_far - current_near;
//                 total_length += interval;
//                 count++;
//                 current_near = -1.0f;
//                 current_far = -1.0f;
//             }
//         }
//     }

//     if (current_near != -1.0f)
//     {
//         float interval = current_far - current_near;
//         total_length += interval;
//         count++;
//         current_near = -1.0f;
//         current_far = -1.0f;
//     }

// }


__device__ inline 
void sample_points_sparse_single_ray(
    float3 origin,
    float3 direction,
    int num_sample,
    float* z_vals, 
    float* dists, 
    float3 block_corner, 
    float3 block_size,
    bool* occupied_gird,
    float near, float far,
    int3 log2dim)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    // printf("bound %f %f\n", bound.x, bound.y);


    if (bound.x == -1) return;

    // if (bound.x == 0.0) bound.x += 0.1f;

    bound.x = fmaxf(bound.x, near);
    // bound.y = fminf(bound.y, far);

    // near = bound.x;
    // far = bound.y;

    int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;


    float temp_len = 0.0f;
    bool flag = false;
    
    // while((!dda.terminate()) && (dda.t.x < far))
    while(!dda.terminate())
    {

        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        // uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = fmax(dda.t.y - dda.t.x, 0.0f);
            flag = true;
            temp_len += len;

        }else{
            if (flag == true) // 上一次是occupied 
            {
                flag = false;
                total_length += temp_len;
                count++;
                temp_len = 0.0f;
            }
        }

        dda.step();
    }

    if (flag == true) // 上一次是occupied 
    {
        flag = false;
        total_length += temp_len;
        count++;
        temp_len = 0.0f;
    }

    // printf("total_length %f count %d\n", total_length, count);


    if (count == 0) return;

    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    int left_sample = num_sample;
    int sample_count = 0;

    float temp_near=0.0f;

    while(!dda.terminate())
    // while((!dda.terminate()) && (dda.t.x < far))
    {
        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;


        if (occupied_gird[n])
        {

            if (flag == false)
            {
                temp_near = dda.t.x;
            }
            float len = fmax(dda.t.y - dda.t.x, 0.0f);
            flag = true;
            temp_len += len;

        }else{
            if (flag == true && temp_len > 0)
            {
                flag = false;

                int num = min(max((int)(num_sample * temp_len / total_length), 1), left_sample);
                if (sample_count == count - 1) 
                {
                    num = left_sample;
                }

                uniform_sample_bound_v3(z_vals+num_sample-left_sample, 
                                        dists+num_sample-left_sample,
                                        temp_near, temp_near+temp_len, num);

                temp_len = 0.0f;
                left_sample = left_sample-num;
                sample_count++;                
            }
        }
        dda.step();
    }

    if (flag == true && temp_len > 0)
    {
        flag = false;

        int num = min(max((int)(num_sample * temp_len / total_length), 1), left_sample);
        if (sample_count == count - 1) 
        {
            num = left_sample;
        }

        uniform_sample_bound_v3(z_vals+num_sample-left_sample, 
                                dists+num_sample-left_sample,
                                temp_near, temp_near+temp_len, num);

        temp_len = 0.0f;
        left_sample = left_sample-num;
        sample_count++;                
    }
}

__device__ inline 
float s_space_to_t_space(float s, float near, float far)
{
    return 1.0f / (1.0f / near * (1.0f - s) + 1.0f / far * s);
}

__device__ inline 
float t_space_to_s_space(float t, float near, float far)
{
    return (1.0f / t - 1.0f / near) / (1.0f / far - 1.0f / near);
}

__device__ inline 
void sample_points_sparse_single_ray_inv_z(
    float3 origin,
    float3 direction,
    int num_sample,
    float* z_vals, 
    float* dists, 
    float3 block_corner, 
    float3 block_size,
    bool* occupied_gird,
    float near, float far,
    int3 log2dim)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    // printf("bound %f %f\n", bound.x, bound.y);


    if (bound.x == -1) return;

    // if (bound.x == 0.0) bound.x += 0.1f;

    bound.x = fmaxf(bound.x, near);
    // bound.y = fminf(bound.y, far);

    near = bound.x;
    far = bound.y;

    int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;


    float temp_len = 0.0f;
    bool flag = false;
    
    // while( (!dda.terminate()) && (dda.t.x < far) )
    while(!dda.terminate())
    {

        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        // uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        if (occupied_gird[n])
        {
            float s_near = t_space_to_s_space(dda.t.x, near, far);
            float s_far = t_space_to_s_space(dda.t.y, near, far);
            float len = fmax(s_far - s_near, 0.0f);
            flag = true;
            temp_len += len;

        }else{
            if (flag == true) // 上一次是occupied 
            {
                flag = false;
                total_length += temp_len;
                count++;
                temp_len = 0.0f;
            }
        }

        dda.step();
    }

    if (flag == true) // 上一次是occupied 
    {
        flag = false;
        total_length += temp_len;
        count++;
        temp_len = 0.0f;
    }

    // printf("total_length %f count %d\n", total_length, count);


    if (count == 0) return;

    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    int left_sample = num_sample;
    int sample_count = 0;

    float temp_near=0.0f;
    float temp_far=0.0f;

    // while( (!dda.terminate()) && (dda.t.x < far) )
    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;


        if (occupied_gird[n])
        {

            if (flag == false)
            {
                temp_near = t_space_to_s_space(dda.t.x, near, far);
                flag = true;
            }
            
            temp_far = t_space_to_s_space(dda.t.y, near, far);

            // float len = fmax(t_space_to_s_space(fmin(dda.t.y, far), near, far) - temp_near, 0.0f);
            
            // temp_len += len;

        }else{
            if (flag == true && temp_far - temp_near > 0)
            {
                flag = false;

                temp_len = temp_far - temp_near;

                int num = min(max((int)(num_sample * temp_len / total_length), 1), left_sample);
                if (sample_count == count - 1) 
                {
                    num = left_sample;
                }

                uniform_sample_bound_v3(z_vals+num_sample-left_sample, 
                                        dists+num_sample-left_sample,
                                        temp_near, temp_far, num);

                temp_len = 0.0f;
                left_sample = left_sample-num;
                sample_count++;                
            }
        }
        dda.step();
    }

    if (flag == true && temp_len > 0)
    {
        flag = false;

        int num = min(max((int)(num_sample * temp_len / total_length), 1), left_sample);
        if (sample_count == count - 1) 
        {
            num = left_sample;
        }

        uniform_sample_bound_v3(z_vals+num_sample-left_sample, 
                                dists+num_sample-left_sample,
                                temp_near, temp_near+temp_len, num);

        temp_len = 0.0f;
        left_sample = left_sample-num;
        sample_count++;                
    }


    for (int i=0; i<num_sample; i++)
    {
        z_vals[i] = s_space_to_t_space(clamp(z_vals[i], 0.0f, 1.0f), near, far);
    }

    for (int i=0; i<num_sample; i++)
    {
        float M  = dists[i] * (1.0f / far - 1.0f / near);
        float t0 = z_vals[i];
        dists[i] = -1.0f  * M * t0 / (M + 1.0f / t0);
    }
    
}

__global__ 
void sample_points_sparse_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample, 
    float* z_vals,
    float* dists,
    float* block_corner,
    float* block_size,
    bool* occupied_gird,
    float near, float far, 
    int log2dim_x, int log2dim_y, int log2dim_z,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        sample_points_sparse_single_ray(rays_o[taskIdx], rays_d[taskIdx], num_sample, 
                                z_vals+taskIdx*num_sample, dists+taskIdx*num_sample,
                                make_float3(block_corner[0], block_corner[1], block_corner[2]), 
                                make_float3(block_size[0],block_size[1],block_size[2]), 
                                occupied_gird, near, far,
                                make_int3(log2dim_x,log2dim_y,log2dim_z));
                                

        taskIdx += total_thread;
    }
}


__global__ 
void sample_points_sparse_inv_z_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample, 
    float* z_vals,
    float* dists,
    float* block_corner,
    float* block_size,
    bool* occupied_gird,
    float near, float far, 
    int log2dim_x, int log2dim_y, int log2dim_z,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        sample_points_sparse_single_ray_inv_z(rays_o[taskIdx], rays_d[taskIdx], num_sample, 
                                z_vals+taskIdx*num_sample, dists+taskIdx*num_sample,
                                make_float3(block_corner[0], block_corner[1], block_corner[2]), 
                                make_float3(block_size[0],block_size[1],block_size[2]), 
                                occupied_gird, near, far,
                                make_int3(log2dim_x,log2dim_y,log2dim_z));
                                

        taskIdx += total_thread;
    }
}


void sample_points_grid(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor &dists,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird,
    float near, float far,
    int log2dim_x, int log2dim_y, int log2dim_z)
{
    int batch_size = rays_o.size(0);
    int num_sample = z_vals.size(1);


    sample_points_sparse_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        num_sample, 
        z_vals.contiguous().data_ptr<float>(),
        dists.contiguous().data_ptr<float>(),
        (float*)block_corner.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(),
        (bool*)occupied_gird.contiguous().data_ptr<bool>(),
        near, far,
        log2dim_x, log2dim_y, log2dim_z,
        batch_size);

    // if (!inv_z)
    // {

    // }else{
    //     sample_points_sparse_inv_z_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
    //         (float3*)rays_o.contiguous().data_ptr<float>(),
    //         (float3*)rays_d.contiguous().data_ptr<float>(),
    //         num_sample, 
    //         z_vals.contiguous().data_ptr<float>(),
    //         dists.contiguous().data_ptr<float>(),
    //         (float*)block_corner.contiguous().data_ptr<float>(),
    //         (float*)block_size.contiguous().data_ptr<float>(),
    //         (bool*)occupied_gird.contiguous().data_ptr<bool>(),
    //         near, far,
    //         log2dim_x, log2dim_y, log2dim_z,
    //         batch_size);
    // }



    AT_CUDA_CHECK(cudaGetLastError());
    return;
}




// occupied grid part 

// __global__ 
// void get_max_value_kernel(
//     float* temp_grid,
//     bool* mask,
//     int* log2dim, 
//     float3* pts, float* value, 
//     int batch_size)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 

//     while(taskIdx < batch_size)
//     {

//         float3 p = pts[taskIdx];

//         float v = value[taskIdx];

//         int3 ijk = make_int3(p);

//         // printf("p %f %f %f ijk %d %d %d\n",
//         // p.x, p.y, p.z, ijk.x, ijk.y, ijk.z);

//         if (ijk.x < 0 || ijk.x >= (1 << log2dim[0]) || 
//             ijk.y < 0 || ijk.y >= (1 << log2dim[1]) ||
//             ijk.z < 0 || ijk.z >= (1 << log2dim[2]))
//         {
//             taskIdx += total_thread;
//             continue;
//         }

//         // int3 resolution = make_int3(1 << log2dim[0], 1<<log2dim[1], 1<<log2dim[3]);
//         // int n = ijk.x * resolution.y * resolution.z + ijk.y * resolution.z + ijk.z;

//         int n = (ijk.x << (log2dim[1] + log2dim[2])) | (ijk.y << log2dim[2]) | ijk.z;
        
//         // printf("p %f %f %f ijk %d %d %d n %f\n",
//         // p.x, p.y, p.z, ijk.x, ijk.y, ijk.z, n);

//         atomicMax(temp_grid + n, v);
//         mask[n] = true;

//         taskIdx += total_thread;
//     }
// }

// __global__ 
// void update_grid_value_kernel(
//     float* temp_grid,
//     bool* mask,
//     float* grid_value,
//     float alpha, 
//     int batch_size)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 

//     while(taskIdx < batch_size)
//     {
//         if (mask[taskIdx])
//         {
//             grid_value[taskIdx] = alpha * grid_value[taskIdx] + 
//                                 (1.0f - alpha) * temp_grid[taskIdx];
//         }

//         taskIdx += total_thread;
//     }
// }


// void update_grid_value_cuda(
//     at::Tensor &grid_value, 
//     at::Tensor log2dim, float alpha, 
//     at::Tensor pts, at::Tensor value)
// {
//     int batch_size = pts.size(0);
//     int total_size = grid_value.size(0)*grid_value.size(1)*grid_value.size(2);

//     float* _temp_grid;
//     cudaMalloc((void**)&_temp_grid, sizeof(float)*total_size);
//     cudaMemset(_temp_grid, 0, sizeof(float)*total_size);
//     bool* _mask;
//     cudaMalloc((void**)&_mask, sizeof(bool)*total_size);
//     cudaMemset(_mask, 0, sizeof(bool)*total_size);

//     get_max_value_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
//         _temp_grid, _mask, (int*)log2dim.contiguous().data_ptr<int>(),
//         (float3*)pts.contiguous().data_ptr<float>(),
//         (float*)value.contiguous().data_ptr<float>(), batch_size);

//     update_grid_value_kernel<<<NUM_BLOCK(total_size), NUM_THREAD>>>(
//         _temp_grid, _mask, (float*)grid_value.contiguous().data_ptr<float>(),
//         alpha, total_size);

//     cudaFree(_temp_grid);
//     cudaFree(_mask);

//     AT_CUDA_CHECK(cudaGetLastError());
//     return;
// }


__global__ 
void padding_inputs_forward_kernel(
    float* inputs, // B x (num_blend * 2) 
    bool* mask, // B x num_blend
    int num_blend,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int channel = num_blend * 2;

        float* cur_inputs = inputs + taskIdx * channel;
        bool* cur_mask = mask + taskIdx * num_blend;

        int count = 0;
        int valid_index[10]; // num_blend should not exceed 10 
        for (int i=0; i<num_blend; i++)
        {
            if (cur_mask[i])
            {
                valid_index[count++] = i;
            }
        }

        if (count == num_blend)
        {
            taskIdx += total_thread;
            continue;
        }

        int cur_index = 0;
        for (int i=0; i<num_blend; i++)
        {
            if (cur_mask[i] == false)
            {
                int index = valid_index[cur_index];

                cur_inputs[i * 2 + 0] = cur_inputs[index * 2 + 0];
                cur_inputs[i * 2 + 1] = cur_inputs[index * 2 + 1];

                cur_index = (cur_index + 1) % count;
            }
        }

        taskIdx += total_thread;
    }
}

void padding_inputs_forward(
    at::Tensor &inputs,
    at::Tensor mask)
{
    int batch_size = inputs.size(0);
    int num_blend = mask.size(1);

    padding_inputs_forward_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        inputs.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        num_blend, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}




__global__ 
void padding_inputs_backward_kernel(
    float* inputs, // B x (num_blend * 2) 
    bool* mask, // B x num_blend
    int num_blend,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int channel = num_blend * 2;

        float* cur_inputs = inputs + taskIdx * channel;
        bool* cur_mask = mask + taskIdx * num_blend;

        int count = 0;
        int valid_index[10]; // num_blend should not exceed 10 
        for (int i=0; i<num_blend; i++)
        {
            if (cur_mask[i])
            {
                valid_index[count++] = i;
            }
        }

        if (count == num_blend)
        {
            taskIdx += total_thread;
            continue;
        }

        int cur_index = 0;
        for (int i=0; i<num_blend; i++)
        {
            if (cur_mask[i] == false)
            {
                int index = valid_index[cur_index];

                cur_inputs[index * 2 + 0] += cur_inputs[i * 2 + 0];
                cur_inputs[index * 2 + 1] += cur_inputs[i * 2 + 1];

                cur_index = (cur_index + 1) % count;
            }
        }

        taskIdx += total_thread;
    }
}

void padding_inputs_backward(
    at::Tensor &inputs,
    at::Tensor mask)
{
    int batch_size = inputs.size(0);
    int num_blend = mask.size(1);

    padding_inputs_backward_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        inputs.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        num_blend, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


// padding depth for invalid depth 
__global__ 
void padding_depth_kernel(
    float* depths, // N x H x W x 1
    bool* valid_mask, // N x H x W x 1
    float* out, // N x H x W x 1
    int num_view,
    int height,
    int width,
    int window_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < num_view * height * width)
    {

        int vidx = taskIdx / (height * width);
        float* cur_depth = depths + vidx * height * width;
        bool* cur_valid_mask = valid_mask + vidx * height * width;
        float* cur_out = out + vidx * height * width;

        int bidx = taskIdx % (height * width);

        int u = bidx % width;
        int v = bidx / width;

        bool valid = cur_valid_mask[bidx];
        
        if (valid)
        {
            cur_out[bidx] = cur_depth[bidx];
            taskIdx += total_thread;
            continue;
        }

        int count = 0;
        float mean_depth = 0.0f;
        int half_size = window_size / 2;
        for (int i=-half_size; i<half_size; i++)
        {
            for (int j=-half_size; j<half_size; j++)
            {
                int x = u + i;
                int y = v + j;
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    if (cur_valid_mask[y * width + x])
                    {
                        mean_depth += cur_depth[y * width + x];
                        count++;
                    }
                }
            }
        }
        if (count > 0) mean_depth = mean_depth / (1.0f * count);

        cur_out[bidx] = mean_depth;

        taskIdx += total_thread;
    }
}


void padding_depth(
    at::Tensor depths, at::Tensor valid_mask,
    at::Tensor &out, int window_size)
{
    int num_view = depths.size(0);
    int height = depths.size(1);
    int width = depths.size(2);
    // printf("Info %d %d %d\n", num_view, height, width);
    padding_depth_kernel<<<NUM_BLOCK(num_view*height*width), NUM_THREAD>>>(
        (float*)depths.contiguous().data_ptr<float>(),
        (bool*)valid_mask.contiguous().data_ptr<bool>(),
        (float*)out.contiguous().data_ptr<float>(),
        num_view, height, width, window_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}



__global__ 
void update_error_map_kernel(
    int* view_idx, // B  
    int* ray_idx,  // B 
    float* error_map, // N x (H/scale) x (W/scale) x  1
    int height, int width,
    int scale, int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    int H = height / scale;
    int W = width / scale;
    while(taskIdx < batch_size)
    {

        int vidx = view_idx[taskIdx];
        int ridx = ray_idx[taskIdx];

        float* cur_error_map = error_map + vidx * H * W;
        

        int u = ridx % width;
        int v = ridx / width;

        int x = u / scale;
        int y = v / scale;


        taskIdx += total_thread;

    }
}



__global__ 
void scatter_value_kernel(
    float* error_map, // num_view x H x W x 1
    float* error, // B x 1
    int* nei_idxs, // B x num_sample x num_neighbor x 1
    float2* warped_uv, // B x num_sample x num_neighbor x 2
    int height, int width,
    int batch_size, int num_sample, int num_neighbor)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size * num_sample * num_neighbor)
    {

        int bidx = taskIdx / (num_sample * num_neighbor);
        // int temp = taskIdx % (num_sample * num_neighbor);
        // int sidx = temp / num_neighbor;
        // int nidx = temp % num_neighbor;

        int nidx = nei_idxs[taskIdx];
        if (nidx == -1)
        {
            taskIdx += total_thread;
            continue;
        }

        float2 uv = warped_uv[taskIdx];
        float e = error[bidx];
        float* cur_error_map = error_map + nidx * height * width;

        int x = (int)(uv.x * width + 0.5f);
        int y = (int)(uv.y * height + 0.5f);

        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            cur_error_map[y * width + x] = e;
        }
            // atomicMax(cur_error_map + y * width + x, e);

        // atomicMax(cur_error_map + y * width + x, e);

        // atomicMax(cur_error_map + y * width + x, e);

        // atomicMax(cur_error_map + y * width + x, e);


        taskIdx += total_thread;
    }
}


void scatter_value_cuda(
    at::Tensor error_map, at::Tensor error,
    at::Tensor nei_idxs, at::Tensor warped_uv)
{
    int height = error_map.size(1);
    int width = error_map.size(2);

    int batch_size = error.size(0);
    int num_sample = nei_idxs.size(1);
    int num_neighbor = nei_idxs.size(2);


    scatter_value_kernel<<<NUM_BLOCK(batch_size*num_sample*num_neighbor), NUM_THREAD>>>(
        (float*)error_map.contiguous().data_ptr<float>(),
        (float*)error.contiguous().data_ptr<float>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(),
        (float2*)warped_uv.contiguous().data_ptr<float>(),
        height, width, batch_size, num_sample, num_neighbor);


    AT_CUDA_CHECK(cudaGetLastError());
    return;

}

__global__ 
void get_focus_kernel(
    float3* rays_o, // B x 3 
    float3* rays_d, // B x 3
    bool* mask, // M1 x M2 x M3 x 1
    int log2dim_x, int log2dim_y, int log2dim_z,
    float* block_corner_arr,
    float* block_size_arr, 
    float near, float far,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 


    int3 log2dim = make_int3(log2dim_x, log2dim_y, log2dim_z);
    float3 block_corner = make_float3(block_corner_arr[0],
                                      block_corner_arr[1],
                                      block_corner_arr[2]);
    float3 block_size = make_float3(block_size_arr[0],
                                    block_size_arr[1],
                                    block_size_arr[2]);
    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];

        float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

        if (bound.x == -1)
        {
            taskIdx += total_thread; continue;
        }

        bound.x = fmaxf(bound.x, near);
        bound.y = fminf(bound.y, far);

        near = bound.x;
        far = bound.y;

        int3 resolution = make_int3(1<<log2dim.x, 1<<log2dim.y, 1<<log2dim.z);
        float3 grid_size = block_size / make_float3(resolution);

        DDASatateScene_v2 dda;
        dda.init(origin-block_corner, direction, bound, resolution, grid_size);

        while( (!dda.terminate()) && (dda.t.x < far) )
        {
            dda.next();
            uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
            mask[n] = true;
            dda.step();
        }
        taskIdx += total_thread;
    }
}

void get_focus_cuda(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor mask,  
    int log2dim_x, int log2dim_y, int log2dim_z,
    at::Tensor block_corner, at::Tensor block_size,
    float near, float far)
{
    int batch_size = rays_d.size(0);

    get_focus_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        log2dim_x, log2dim_y, log2dim_z,
        (float*)block_corner.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(), 
        near, far,
        batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}



// ============================================= GRID =====================================//

/// @brief Set all bits off
template<uint32_t SIZE>
static inline __device__ void setOff(uint64_t* mWords)
{
    for (uint32_t i = 0; i < (SIZE >> 6); ++i)
        mWords[i] = uint64_t(0);
}

/// @brief Set the specified bit off.
static inline __device__ void setOff(uint32_t n, uint64_t* mWords) 
{ 
    uint64_t temp = ~(uint64_t(1) << (n & 63));
    n = n >> 6;
    atomicAnd((uint64_cu*)mWords+n, (uint64_cu)temp);
    // mWords[n >> 6] &= ~(uint64_t(1) << (n & 63)); 
}

/// @brief Set the specified bit on.
static inline __device__ void setOn(uint32_t n, uint64_t* mWords) 
{ 
    uint64_t temp = uint64_t(1) << (n & 63);
    n = n >> 6;
    atomicOr((uint64_cu*)mWords+n, (uint64_cu)temp);
    // mWords[n >> 6] |= uint64_t(1) << (n & 63); 
}

__global__ 
void seton_voxel_grid_kernel(
    float3* pts, // B x 3 ijk locate 
    float* value, // B x 1
    VoxelGrid grid, 
    VoxelGrid visited, 
    float thresh, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    int3 grid_resolution = make_int3(1 << grid.log2dim_x, 1 << grid.log2dim_y, 1 << grid.log2dim_z);

    while(taskIdx < batch_size)
    {

        float3 p = pts[taskIdx]; // contracted point

        // printf("grid_resolution %d %d %d", grid_resolution.x, grid_resolution.y, grid_resolution.z);
        int3 ijk = make_int3( (int)((p.x + 2.0f) / 4.0f * grid_resolution.x),
                                (int)((p.y + 2.0f) / 4.0f * grid_resolution.y),
                                (int)((p.z + 2.0f) / 4.0f * grid_resolution.z));
        // printf("i %d j %d k %d\n", ijk.x, ijk.y, ijk.z);

        if (value[taskIdx] >= thresh)
        {

            // setOn(visited.ijk2offset(ijk.x, ijk.y, ijk.z), visited.data);

            // visited.isOn(ijk.x, ijk.y, ijk.z)
            visited.setOn(ijk.x, ijk.y, ijk.z);

            setOn(grid.ijk2offset(ijk.x, ijk.y, ijk.z), grid.data);
            // grid.setOn(ijk.x, ijk.y, ijk.z);
        }

        taskIdx += total_thread;
    }
}

__global__ 
void setoff_voxel_grid_kernel(
    float3* pts, // B x 3 ijk locate 
    float* value, // B x 1
    VoxelGrid grid, 
    VoxelGrid visited, 
    float thresh, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    int3 grid_resolution = make_int3(1 << grid.log2dim_x, 1 << grid.log2dim_y, 1 << grid.log2dim_z);

    while(taskIdx < batch_size)
    {

        float3 p = pts[taskIdx]; // contracted point

        // printf("grid_resolution %d %d %d", grid_resolution.x, grid_resolution.y, grid_resolution.z);
        int3 ijk = make_int3( (int)((p.x + 2.0f) / 4.0f * grid_resolution.x),
                                (int)((p.y + 2.0f) / 4.0f * grid_resolution.y),
                                (int)((p.z + 2.0f) / 4.0f * grid_resolution.z));
        // printf("i %d j %d k %d\n", ijk.x, ijk.y, ijk.z);

        if (value[taskIdx] < thresh && visited.isOff(ijk.x, ijk.y, ijk.z))
        {
            setOff(grid.ijk2offset(ijk.x, ijk.y, ijk.z), grid.data);
        }

        taskIdx += total_thread;
    }
}


void update_voxel_grid_cuda(
    at::Tensor pts, at::Tensor value,
    at::Tensor grid_data, 
    int log2dim_x, int log2dim_y, int log2dim_z,
    float thresh)
{
    int batch_size = pts.size(0);
    // printf("log2dim %d %d %d", log2dim_x, log2dim_y, log2dim_z);
    VoxelGrid grid(
        (uint64_t*)grid_data.contiguous().data_ptr<int64_t>(),
        log2dim_x, log2dim_y, log2dim_z);


    uint64_t total_size = 1 << (log2dim_x + log2dim_y + log2dim_z);
    uint64_t* _temp_grid;
    cudaMalloc((void**)&_temp_grid, sizeof(uint64_t)*total_size);
    cudaMemset(_temp_grid, 0, sizeof(uint64_t)*total_size);

    VoxelGrid visited(_temp_grid, log2dim_x, log2dim_y, log2dim_z);
    

    seton_voxel_grid_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(),
        (float*)value.contiguous().data_ptr<float>(), 
        grid, visited, thresh, batch_size);

    setoff_voxel_grid_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(),
        (float*)value.contiguous().data_ptr<float>(), 
        grid, visited, thresh, batch_size);
    
    cudaFree(_temp_grid);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


__global__ 
void seton_all_kernel(
    VoxelGrid grid, 
    uint64_t total_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < total_size)
    {
        setOn(taskIdx, grid.data);
        taskIdx += total_thread;
    }
}

__global__ 
void setoff_all_kernel(
    VoxelGrid grid, 
    uint64_t total_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < total_size)
    {
        setOff(taskIdx, grid.data);
        taskIdx += total_thread;
    }
}


void set_all_cuda(
    at::Tensor grid_data,
    int log2dim_x, int log2dim_y, int log2dim_z,
    bool flag)
{
    uint64_t total_size = 1 << (log2dim_x + log2dim_y + log2dim_z);
    VoxelGrid grid(
        (uint64_t*)grid_data.contiguous().data_ptr<int64_t>(),
        log2dim_x, log2dim_y, log2dim_z);
    // printf("total_size %ld\n", total_size);
    if (flag)
    {
        seton_all_kernel<<<NUM_BLOCK(total_size), NUM_THREAD>>>(grid, total_size);
    }else{
        setoff_all_kernel<<<NUM_BLOCK(total_size), NUM_THREAD>>>(grid, total_size);
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


__global__ 
void voxel_grid_to_mask_kernel(
    VoxelGrid grid, 
    bool* mask, 
    uint64_t total_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < total_size)
    {
        mask[taskIdx] = grid.isOn((uint32_t)taskIdx);
        taskIdx += total_thread;
    }
}

void voxel_grid_to_mask_cuda(
    at::Tensor grid_data, 
    int log2dim_x, int log2dim_y, int log2dim_z,
    at::Tensor mask)
{
    uint64_t total_size = 1 << (log2dim_x + log2dim_y + log2dim_z);


    VoxelGrid grid(
        (uint64_t*)grid_data.contiguous().data_ptr<int64_t>(),
        log2dim_x, log2dim_y, log2dim_z);

    voxel_grid_to_mask_kernel<<<NUM_BLOCK(total_size), NUM_THREAD>>>(
        grid, (bool*)mask.contiguous().data_ptr<bool>(), total_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}

__global__ 
void split_voxel_grid_kernel(
    VoxelGrid old_grid, 
    VoxelGrid new_grid, uint64_t total_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < total_size)
    {

        uint3 ijk = new_grid.offset2ijk(taskIdx);

        int old_i = ijk.x / 2;
        int old_j = ijk.y / 2;
        int old_k = ijk.z / 2;

        bool flag = old_grid.isOn(old_i, old_j, old_k);

        if (flag)
        {
            setOn(new_grid.ijk2offset((int)ijk.x, (int)ijk.y, (int)ijk.z), new_grid.data);
            // new_grid.setOn((int)ijk.x, (int)ijk.y, (int)ijk.z);
        }
        // }else{
        //     setOff(new_grid.ijk2offset((int)ijk.x, (int)ijk.y, (int)ijk.z), new_grid.data);
        //     // new_grid.setOff((int)ijk.x, (int)ijk.y, (int)ijk.z);
        // }

        taskIdx += total_thread;
    }
}

void split_voxel_grid_cuda(
    at::Tensor grid_data, at::Tensor new_grid_data,
    int log2dim_x, int log2dim_y, int log2dim_z)
{

    uint64_t total_size = 1 << (log2dim_x + log2dim_y + log2dim_z);
    
    VoxelGrid old_grid(
        (uint64_t*)grid_data.contiguous().data_ptr<int64_t>(),
        log2dim_x-1, log2dim_y-1, log2dim_z-1);

    VoxelGrid new_grid(
        (uint64_t*)new_grid_data.contiguous().data_ptr<int64_t>(),
        log2dim_x, log2dim_y, log2dim_z);

    split_voxel_grid_kernel<<<NUM_BLOCK(total_size), NUM_THREAD>>>(
        old_grid, new_grid, total_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}

// __device__ 
// void sparse_voxel_contracted_grid_sampling(
//     float3 origin, float3 direction,
//     float near, float far, //  given from data
//     int num_sample, 
//     VoxelGrid grid, 
//     float* s_vals,
//     float* s_dists)
// {
//     // origin already inside [-1,1]^3
//     // block center, block_size world space [-2,2]^2 block_size = 4

//     // 由于 sparse voxel grid, 末尾的t_vals可能没有采样点

//     float inv_n = 1.0f / near;
//     float inv_f = 1.0f / far;

//     // s space 
//     float ds = 1.0f / num_sample;
//     float s = 0.0f;

//     int count = 0;
//     int3 grid_resolution = make_int3(1 << grid.log2dim_x, 1 << grid.log2dim_y, 1 << grid.log2dim_z);

//     while (s < 1.0f && count < num_sample)
//     {

//         float t = 1.0f / (inv_n * (1.0f - s) + inv_f * s);
//         float3 pts = origin + t * direction; // world space points 

//         // check its occ in grid 
//         float3 c_pts = pts; // contracted points 

//         float mag = fmaxf(fabsf(pts.x), fmaxf(fabsf(pts.y), fabsf(pts.z)));

//         if (mag > 1)
//         {
//             // [-2,2]
//             float Linfscale = (2 - 1 / mag) / mag;
//             c_pts.x *= Linfscale;
//             c_pts.y *= Linfscale;
//             c_pts.z *= Linfscale;
//         }

//         int3 n_pts = make_int3( (int)((c_pts.x + 2.0f) / 4.0f * grid_resolution.x),
//                                 (int)((c_pts.y + 2.0f) / 4.0f * grid_resolution.y),
//                                 (int)((c_pts.z + 2.0f) / 4.0f * grid_resolution.z));

//         if (grid.isOn(n_pts.x, n_pts.y, n_pts.z))
//         {
//             // [FIXME] here
//             s_vals[0] = t;
//             s_dists[0] = ds;
//             s_vals += 1;
//             s_dists += 1;
//             count++;
//         }
//         s = s + ds;
//     }
// }

// __global__ 
// void sparse_voxel_contracted_grid_sampling_kernel(
//     float3* rays_o, float3* rays_d,
//     float near, float far, int num_sample,
//     VoxelGrid grid, 
//     float* s_vals, float* s_dists, int batch_size)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 
//     while(taskIdx < batch_size)
//     {
//         sparse_voxel_contracted_grid_sampling(
//             rays_o[taskIdx], rays_d[taskIdx],
//             near, far, num_sample, grid, 
//             s_vals + taskIdx * num_sample,
//             s_dists + taskIdx * num_sample);
//         taskIdx += total_thread;
//     }
// }

// void sparse_voxel_contracted_grid_sampling_cuda(
//     at::Tensor rays_o, at::Tensor rays_d, 
//     float near, float far, at::Tensor grid_data,
//     int log2dim_x, int log2dim_y, int log2dim_z,
//     at::Tensor s_vals, at::Tensor s_dists)
// {
//     int batch_size = rays_o.size(0);
//     int num_sample = s_vals.size(1);

//     VoxelGrid grid(
//         (uint64_t*)grid_data.contiguous().data_ptr<int64_t>(),
//         log2dim_x, log2dim_y, log2dim_z);


//     sparse_voxel_contracted_grid_sampling_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
//         (float3*)rays_o.contiguous().data_ptr<float>(),
//         (float3*)rays_d.contiguous().data_ptr<float>(),
//         near, far, num_sample, grid, 
//         (float*)s_vals.contiguous().data_ptr<float>(),
//         (float*)s_dists.contiguous().data_ptr<float>(), batch_size);

//     AT_CUDA_CHECK(cudaGetLastError());
//     return;

// }


__global__ 
void get_max_value_kernel(
    float* temp_grid,
    bool* mask,
    int log2dim_x, int log2dim_y, int log2dim_z,
    float3* pts, float* value, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    
    int3 resolution = make_int3(1<<log2dim_x, 1<<log2dim_y, 1<<log2dim_z); 

    while(taskIdx < batch_size)
    {

        float3 p = pts[taskIdx];

        float v = value[taskIdx];

        // int3 ijk = make_int3(p);

        if (p.x <= 0 || p.x >= 1 ||
            p.y <= 0 || p.y >= 1 ||
            p.z <= 0 || p.z >= 1)
        {
            taskIdx += total_thread;
            continue;
        }

        int3 ijk = make_int3( (int)(p.x * resolution.x),
                              (int)(p.y * resolution.y),
                              (int)(p.z * resolution.z));

        ijk.x = clamp(ijk.x, 0, resolution.x-1);
        ijk.y = clamp(ijk.y, 0, resolution.y-1);
        ijk.z = clamp(ijk.z, 0, resolution.z-1);

        // printf("p %f %f %f ijk %d %d %d\n",
        // p.x, p.y, p.z, ijk.x, ijk.y, ijk.z);

        // if (ijk.x < 0 || ijk.x >= (1 << log2dim[0]) || 
        //     ijk.y < 0 || ijk.y >= (1 << log2dim[1]) ||
        //     ijk.z < 0 || ijk.z >= (1 << log2dim[2]))
        // {
        //     taskIdx += total_thread;
        //     continue;
        // }

        // int3 resolution = make_int3(1 << log2dim[0], 1<<log2dim[1], 1<<log2dim[3]);
        // int n = ijk.x * resolution.y * resolution.z + ijk.y * resolution.z + ijk.z;

        int n = (ijk.x << (log2dim_y + log2dim_z)) | (ijk.y << log2dim_z) | ijk.z;
        
        // printf("p %f %f %f ijk %d %d %d n %f\n",
        // p.x, p.y, p.z, ijk.x, ijk.y, ijk.z, n);

        atomicMax(temp_grid + n, v);
        mask[n] = true;

        taskIdx += total_thread;
    }
}

__global__ 
void update_grid_value_kernel(
    float* temp_grid,
    bool* mask,
    float* grid_value,
    float alpha, 
    uint64_t batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        if (mask[taskIdx])
        {
            grid_value[taskIdx] = alpha * grid_value[taskIdx] + 
                                (1.0f - alpha) * temp_grid[taskIdx];
        }

        taskIdx += total_thread;
    }
}


void update_grid_value_cuda(
    at::Tensor &grid_value, 
    int log2dim_x, int log2dim_y, int log2dim_z, float alpha, 
    at::Tensor pts, at::Tensor value)
{
    int batch_size = pts.size(0);
    uint64_t total_size = grid_value.size(0)*grid_value.size(1)*grid_value.size(2);

    float* _temp_grid;
    cudaMalloc((void**)&_temp_grid, sizeof(float)*total_size);
    cudaMemset(_temp_grid, 0, sizeof(float)*total_size);
    bool* _mask;
    cudaMalloc((void**)&_mask, sizeof(bool)*total_size);
    cudaMemset(_mask, 0, sizeof(bool)*total_size);

    get_max_value_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        _temp_grid, _mask, log2dim_x, log2dim_y, log2dim_z,
        (float3*)pts.contiguous().data_ptr<float>(),
        (float*)value.contiguous().data_ptr<float>(), batch_size);

    update_grid_value_kernel<<<NUM_BLOCK(total_size), NUM_THREAD>>>(
        _temp_grid, _mask, (float*)grid_value.contiguous().data_ptr<float>(),
        alpha, total_size);

    cudaFree(_temp_grid);
    cudaFree(_mask);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


__device__ 
void sparse_voxel_contracted_grid_sampling(
    float3 origin, float3 direction,
    float near, float far, //  given from data
    int num_sample, 
    bool* grid_mask, 
    int log2dim_x, int log2dim_y, int log2dim_z, 
    float* s_vals,
    float* s_dists)
{
    // origin already inside [-1,1]^3
    // block center, block_size world space [-2,2]^2 block_size = 4

    // 由于 sparse voxel grid, 末尾的t_vals可能没有采样点

    float inv_n = 1.0f / near;
    float inv_f = 1.0f / far;

    // s space 
    float ds = 1.0f / num_sample;
    float s = 0.0f;

    int count = 0;
    int3 grid_resolution = make_int3(1 << log2dim_x, 1 << log2dim_y, 1 << log2dim_z);

    while (s < 1.0f && count < num_sample)
    {

        float t = 1.0f / (inv_n * (1.0f - s) + inv_f * s);
        float3 pts = origin + t * direction; // world space points 

        // check its occ in grid 
        float3 c_pts = pts; // contracted points 

        float mag = fmaxf(fabsf(pts.x), fmaxf(fabsf(pts.y), fabsf(pts.z)));

        if (mag > 1)
        {
            // [-2,2]
            float Linfscale = (2 - 1 / mag) / mag;
            c_pts.x *= Linfscale;
            c_pts.y *= Linfscale;
            c_pts.z *= Linfscale;
        }

        int3 n_pts = make_int3( (int)((c_pts.x + 2.0f) / 4.0f * grid_resolution.x),
                                (int)((c_pts.y + 2.0f) / 4.0f * grid_resolution.y),
                                (int)((c_pts.z + 2.0f) / 4.0f * grid_resolution.z));

        int n = (n_pts.x << (log2dim_y + log2dim_z)) | (n_pts.y << log2dim_z) | n_pts.z;
        if (grid_mask[n])
        {
            // [FIXME] here
            s_vals[0] = t;
            s_dists[0] = ds;
            s_vals += 1;
            s_dists += 1;
            count++;
        }
        s = s + ds;
    }
}


__global__ 
void sparse_voxel_contracted_grid_sampling_kernel(
    float3* rays_o, float3* rays_d,
    float near, float far, int num_sample,
    bool* grid_mask, 
    int log2dim_x, int log2dim_y, int log2dim_z, 
    float* s_vals, float* s_dists, int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        sparse_voxel_contracted_grid_sampling(
            rays_o[taskIdx], rays_d[taskIdx],
            near, far, num_sample, grid_mask, 
            log2dim_x, log2dim_y, log2dim_z,
            s_vals + taskIdx * num_sample,
            s_dists + taskIdx * num_sample);
        taskIdx += total_thread;
    }
}

void sparse_voxel_contracted_grid_sampling_cuda(
    at::Tensor rays_o, at::Tensor rays_d, 
    float near, float far, at::Tensor grid_mask,
    int log2dim_x, int log2dim_y, int log2dim_z,
    at::Tensor s_vals, at::Tensor s_dists)
{
    int batch_size = rays_o.size(0);
    int num_sample = s_vals.size(1);



    sparse_voxel_contracted_grid_sampling_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        near, far, num_sample,
        (bool*)grid_mask.contiguous().data_ptr<bool>(),
        log2dim_x, log2dim_y, log2dim_z,
        (float*)s_vals.contiguous().data_ptr<float>(),
        (float*)s_dists.contiguous().data_ptr<float>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}



// process invalid cases 
__global__ 
void invalid_find_guide_kernel(
    bool* valid_mask, // num_view x H x W x 1
    bool* valid, // B x 1
    int* view_idxs, // B x 1 
    int* ray_idxs, // B x 1
    float2* guide_uv, // B x 2
    int max_range, 
    int num_view,
    int height, int width,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        if (valid[taskIdx]==false)
        {
            int vidx = view_idxs[taskIdx];
            int ridx = ray_idxs[taskIdx];

            bool* cur_valid_mask = valid_mask + vidx * height * width;

            int x = ridx % width;
            int y = ridx / width;

            // find the nearest valid neighbor 
            for (int i=0; i<max_range; i++)
            {
                if (x - i >= 0)
                {
                    int idx = y * width + x - i;
                    if (cur_valid_mask[idx])
                    {
                        guide_uv[taskIdx] = make_float2( x - i, y );
                        break;
                    }
                }

                if (x + i < width)
                {
                    int idx = y * width + x + i;
                    if (cur_valid_mask[idx])
                    {
                        guide_uv[taskIdx] = make_float2( x + i, y );
                        break;
                    }
                }

                if (y - i >= 0)
                {
                    int idx = (y - i) * width + x;
                    if (cur_valid_mask[idx])
                    {
                        guide_uv[taskIdx] = make_float2( x, y - i);
                        break;
                    }
                }

                if (y + i < height)
                {
                    int idx = (y + i) * width + x;
                    if (cur_valid_mask[idx])
                    {
                        guide_uv[taskIdx] = make_float2( x, y + i);
                        break;
                    }
                }
            }
        }

        taskIdx += total_thread;
    }
}


void invalid_find_guide_cuda(
    at::Tensor valid_mask, at::Tensor valid,
    at::Tensor view_idxs, at::Tensor ray_idxs,
    at::Tensor &guide_uv, int max_range)
{
    int num_view = valid_mask.size(0);
    int height = valid_mask.size(1);
    int width = valid_mask.size(2);
    int batch_size = valid.size(0);

    invalid_find_guide_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (bool*)valid_mask.contiguous().data_ptr<bool>(),
        (bool*)valid.contiguous().data_ptr<bool>(),
        (int*)view_idxs.contiguous().data_ptr<int>(),
        (int*)ray_idxs.contiguous().data_ptr<int>(),
        (float2*)guide_uv.contiguous().data_ptr<float>(),
        max_range, num_view, height, width, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}



__global__ 
void get_warping_mask_kernel(
    int* ref_idxs, // B x 1
    float3* rays_o,  // B x 3
    float3* rays_d,  // B x 3
    float3* samples,  // B x num_sample x 3
    PinholeCameraManager cameras, 
    int* candidate_neighbors, // B x num_candidate
    bool* mask, // B x num_sample x num_candidate x 1
    int height, int width,
    int num_sample,
    int num_camera,
    int num_candidate,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;

        int ref_idx = ref_idxs[batch_idx];
        int neighbor_idx = candidate_neighbors[batch_idx * num_candidate + blockIdx.y];

        if (ref_idx == neighbor_idx)
        {
            taskIdx += total_thread;
            continue;
        }

        float3 origin = rays_o[batch_idx];
        float3 direction = rays_d[batch_idx];
        float3 x_world = samples[taskIdx];
        
        PinholeCamera cam = cameras[neighbor_idx];

        float3 x_cam = cam.world2cam(x_world);
        
        float3 x_pixel = cam.cam2pixel(x_cam);

        if (x_pixel.z <= 0)
        {
            taskIdx += total_thread;
            continue;
        }

        float2 uv = make_float2(x_pixel.x / x_pixel.z, x_pixel.y / x_pixel.z);
        if (uv.x < 0 || uv.y < 0 || uv.x >= width || uv.y >= height)
        {
            taskIdx += total_thread;
            continue;
            // out_view_port = true;
        }

        mask[taskIdx * num_candidate + blockIdx.y] = true;

        taskIdx += total_thread;
    }
}

void get_warping_mask_cuda(
    at::Tensor ref_idxs,
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor samples, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor candidate_neighbors,
    at::Tensor mask,
    int height, int width)
{
    int batch_size = samples.size(0);
    int num_sample = samples.size(1);
    int num_camera = rts.size(0);
    int num_candidate = candidate_neighbors.size(1);

    PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);


    get_warping_mask_kernel<<<dim3(NUM_BLOCK(batch_size*num_sample), num_candidate), NUM_THREAD>>>(
        (int*)ref_idxs.contiguous().data_ptr<int>(),
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)samples.contiguous().data_ptr<float>(), cameras,
        (int*)candidate_neighbors.contiguous().data_ptr<int>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        height, width, num_sample, num_camera, num_candidate, batch_size);


    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


__global__ 
void padding_results_kernel(
    bool* valid, // B x 1
    float3* colors, // B x 3
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < height*width)
    {
        if (valid[taskIdx] == false)
        {
            int x = taskIdx % width;
            int y = taskIdx / width;

            float2 uv = nearest_valid_neighbor_search(valid, height, width, x, y, 200);

            if (uv.x > 0)
            {
                colors[taskIdx] = colors[(int)uv.y * width + (int)uv.x];
            }
            
        }
        taskIdx += total_thread;
    }
}

void padding_results_cuda(
    at::Tensor valid,
    at::Tensor img)
{
    int height = img.size(0);
    int width = img.size(1);

    padding_results_kernel<<<NUM_BLOCK(height*width), NUM_THREAD>>>(
        (bool*)valid.contiguous().data_ptr<bool>(),
        (float3*)img.contiguous().data_ptr<float>(),
        height, width);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}

__global__ 
void grid_sample_nearest_forward_kernel(
    float* src, // N x C x H x W
    float2* uvs, // B x 2
    float* output, // B x C 
    int* view_idxs, //  B x 1
    int num_channel, 
    int height, int width,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size * num_channel)
    {
        int bidx = taskIdx / num_channel;
        int channel = taskIdx % num_channel;

        int idx = view_idxs[bidx];

        if (idx == -1)
        {
            taskIdx += total_thread;
            continue;   
        }

        float2 uv = uvs[bidx]; // [0, 1]
        // uv = clamp(uv, 0.0f, 1.0f);

        float* cur_src = src + (idx * num_channel + channel) * height * width;

        int x = clamp(int(uv.x * width), 0, width-1);
        int y = clamp(int(uv.y * height), 0, height-1);
        
        output[taskIdx] = cur_src[y * width + x];

        taskIdx += total_thread;
    }
}

__global__ 
void grid_sample_nearest_backward_kernel(
    float* grad_src, // N x C x H x W
    float2* uvs, // B x 2
    float* grad_in, // B x C
    int* view_idxs, //  B x 1
    int num_channel, 
    int height, int width,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size * num_channel)
    {
        int bidx = taskIdx / num_channel;
        int channel = taskIdx % num_channel;

        int idx = view_idxs[bidx];

        if (idx == -1)
        {
            taskIdx += total_thread;
            continue;   
        }

        float2 uv = uvs[bidx]; // [0, 1]
        // uv = clamp(uv, 0.0f, 1.0f);

        float* cur_grad_src = grad_src + (idx * num_channel + channel) * height * width;

        int x = clamp(int(uv.x * width), 0, width-1);
        int y = clamp(int(uv.y * height), 0, height-1);

        atomicAdd(cur_grad_src+y * width + x, grad_in[taskIdx]);

        taskIdx += total_thread;
    }
}


void grid_sample_nearest(
    at::Tensor src, at::Tensor uv,
    at::Tensor out, at::Tensor vidx,
    bool forward)
{
    int num_channel = src.size(1);
    int height = src.size(2);
    int width = src.size(3);
    int batch_size = uv.size(0);

    if (forward)
    {
        grid_sample_nearest_forward_kernel<<<NUM_BLOCK(batch_size * num_channel), NUM_THREAD>>>(
            (float*)src.contiguous().data_ptr<float>(),
            (float2*)uv.contiguous().data_ptr<float>(),
            (float*)out.contiguous().data_ptr<float>(),
            (int*)vidx.contiguous().data_ptr<int>(),
            num_channel, height, width, batch_size);
    }else{
        grid_sample_nearest_backward_kernel<<<NUM_BLOCK(batch_size * num_channel), NUM_THREAD>>>(
            (float*)src.contiguous().data_ptr<float>(),
            (float2*)uv.contiguous().data_ptr<float>(),
            (float*)out.contiguous().data_ptr<float>(),
            (int*)vidx.contiguous().data_ptr<int>(),
            num_channel, height, width, batch_size);
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return;

}