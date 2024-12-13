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


// __device__ float fetch_depth(float* src, float2 uv, int height, int width)
// {
//     // src H x W x 3 
//     uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
//     uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

//     if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
//     {
//         return 0.0f;
//     }

//     int x0 = (int)uv.x;
//     int y0 = (int)uv.y;
//     int x1 = x0 + 1;
//     int y1 = y0 + 1;
//     float x = uv.x - x0;
//     float y = uv.y - y0;

//     int idx00 = x0 + y0 * width;
//     int idx01 = idx00 + width;
//     int idx10 = idx00 + 1;
//     int idx11 = idx01 + 1;

//     return src[idx00] * (1.0f - x) * (1.0f - y) + 
//            src[idx01] * (1.0f - x) * y + 
//            src[idx10] * x * (1.0f - y) + 
//            src[idx11] * x * y;
// }


// __device__ 
// inline float gradient_of_fetch_depth(float* src, float2 uv, int height, int width, 
//                                      float grid_in)
// {
//     // src H x W x 3 
//     uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
//     uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

//     if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
//     {
//         return 0.0f;
//     }

//     int x0 = (int)uv.x;
//     int y0 = (int)uv.y;
//     int x1 = x0 + 1;
//     int y1 = y0 + 1;
//     float x = uv.x - x0;
//     float y = uv.y - y0;

//     int idx00 = x0 + y0 * width;
//     int idx01 = idx00 + width;
//     int idx10 = idx00 + 1;
//     int idx11 = idx01 + 1;


//     float3 v00 = make_float3((float)src[idx00 * 3 + 0], 
//                              (float)src[idx00 * 3 + 1],
//                              (float)src[idx00 * 3 + 2]);
//     float3 v01 = make_float3((float)src[idx01 * 3 + 0], 
//                              (float)src[idx01 * 3 + 1],
//                              (float)src[idx01 * 3 + 2]);
//     float3 v10 = make_float3((float)src[idx10 * 3 + 0], 
//                              (float)src[idx10 * 3 + 1],
//                              (float)src[idx10 * 3 + 2]);
//     float3 v11 = make_float3((float)src[idx11 * 3 + 0], 
//                              (float)src[idx11 * 3 + 1],
//                              (float)src[idx11 * 3 + 2]);
    
//     float3 grad_x = (-1.0f * v00 * (1.0f - y) - v01 * y + v10 * (1.0f - y) + v11 * y) * (width-1.0f) / 2.0f;

//     float3 grad_y = (-1.0f * v00 * (1.0f - x) + v01 * (1.0f - x) - v10 * x + v11 * x) * (height-1.0f) / 2.0f;

    
//     return make_float2(dot(grid_in,grad_x), dot(grid_in, grad_y));
// }


__device__ 
inline float3 fetch_color(uint8_t* src, float2 uv, int height, int width)
{
    // src H x W x 3 
    uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
    uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

    if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
    {
        return make_float3(-1,-1,-1);
    }

    int x0 = (int)uv.x;
    int y0 = (int)uv.y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float x = uv.x - x0;
    float y = uv.y - y0;

    int idx00 = x0 + y0 * width;
    int idx01 = idx00 + width;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;

    float3 v00 = make_float3((float)src[idx00 * 3 + 0], 
                             (float)src[idx00 * 3 + 1],
                             (float)src[idx00 * 3 + 2]);
    float3 v01 = make_float3((float)src[idx01 * 3 + 0], 
                             (float)src[idx01 * 3 + 1],
                             (float)src[idx01 * 3 + 2]);
    float3 v10 = make_float3((float)src[idx10 * 3 + 0], 
                             (float)src[idx10 * 3 + 1],
                             (float)src[idx10 * 3 + 2]);
    float3 v11 = make_float3((float)src[idx11 * 3 + 0], 
                             (float)src[idx11 * 3 + 1],
                             (float)src[idx11 * 3 + 2]);
    

    return v00 * (1.0f - x) * (1.0f - y) + 
           v01 * (1.0f - x) * y + 
           v10 * x * (1.0f - y) + 
           v11 * x * y;
}

__device__ 
inline float2 gradient_of_fetch_color(uint8_t* src, float2 uv, int height, int width, 
                                      float3 grid_in)
{
    // src H x W x 3 
    uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
    uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

    if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
    {
        return make_float2(0,0);
    }

    int x0 = (int)uv.x;
    int y0 = (int)uv.y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float x = uv.x - x0;
    float y = uv.y - y0;

    int idx00 = x0 + y0 * width;
    int idx01 = idx00 + width;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;


    float3 v00 = make_float3((float)src[idx00 * 3 + 0], 
                             (float)src[idx00 * 3 + 1],
                             (float)src[idx00 * 3 + 2]);
    float3 v01 = make_float3((float)src[idx01 * 3 + 0], 
                             (float)src[idx01 * 3 + 1],
                             (float)src[idx01 * 3 + 2]);
    float3 v10 = make_float3((float)src[idx10 * 3 + 0], 
                             (float)src[idx10 * 3 + 1],
                             (float)src[idx10 * 3 + 2]);
    float3 v11 = make_float3((float)src[idx11 * 3 + 0], 
                             (float)src[idx11 * 3 + 1],
                             (float)src[idx11 * 3 + 2]);
    
    float3 grad_x = (-1.0f * v00 * (1.0f - y) - v01 * y + v10 * (1.0f - y) + v11 * y) * (width-1.0f) / 2.0f;

    float3 grad_y = (-1.0f * v00 * (1.0f - x) + v01 * (1.0f - x) - v10 * x + v11 * x) * (height-1.0f) / 2.0f;

    
    return make_float2(dot(grid_in,grad_x), dot(grid_in, grad_y));
}


__global__ void grid_sample_forward_kernel(
    uint8_t* src, // N x H x W x 3 
    float2* grid, // N x B x 1 x 2
    float3* out, // N x B x 1 x 3
    bool* mask, // N x B x 1 x 1
    int num_imgaes,
    int batch_size, 
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int loc = blockIdx.y * batch_size + taskIdx;

        float2 uv = grid[loc];
        
        float3 color = fetch_color(src + blockIdx.y * (height * width * 3), uv, height, width);

        if (color.x == -1)
        {
            mask[loc] = false;
            color = make_float3(0,0,0);
        }
        else mask[loc] = true;

        out[loc] = color;

        taskIdx += total_thread;
    }
}


void grid_sample_forward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor out,
    at::Tensor mask)
{
    int num_imgaes = src.size(0);
    int batch_size = grid.size(1);
    int height = src.size(1);
    int width = src.size(2);

    dim3 num_block = dim3(NUM_BLOCK(batch_size), num_imgaes);
    grid_sample_forward_kernel<<<num_block, NUM_THREAD>>>(
        (uint8_t*)src.contiguous().data_ptr<uint8_t>(),
        (float2*)grid.contiguous().data_ptr<float>(),
        (float3*)out.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        num_imgaes, batch_size, height, width);
    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}

__global__ void grid_sample_backward_kernel(
    uint8_t* src, // N x H x W x 3 
    float2* grid, // N x B x 1 x 2
    float3 * grad_in, //  N x B x 1 x 3
    float2* grad_grid, // N x B x 1 x 2
    int num_imgaes,
    int batch_size, 
    int height, int width){

    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int loc = blockIdx.y * batch_size + taskIdx;

        float2 uv = grid[loc];
        
        grad_grid[loc] = gradient_of_fetch_color(src + blockIdx.y * (height * width * 3), 
                                                uv, height, width, grad_in[loc]);

        taskIdx += total_thread;
    }
}


void grid_sample_backward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor grad_in,
    at::Tensor grad_grid)
{
    int num_imgaes = src.size(0);
    int batch_size = grid.size(1);
    int height = src.size(1);
    int width = src.size(2);

    dim3 num_block = dim3(NUM_BLOCK(batch_size), num_imgaes);
    grid_sample_backward_kernel<<<num_block, NUM_THREAD>>>(
        (uint8_t*)src.contiguous().data_ptr<uint8_t>(),
        (float2*)grid.contiguous().data_ptr<float>(),
        (float3*)grad_in.contiguous().data_ptr<float>(),
        (float2*)grad_grid.contiguous().data_ptr<float>(),
        num_imgaes, batch_size, height, width);
    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__device__ 
inline float3 gaussian_interpolate_forward(uint8_t* src, float2 uv, int height, int width, float sigma, float max_dis)
{
    /*
    sigma  高斯函数的sigma 
    max_dis  最大距离
    */

    float item = -1.0f / (sigma * sigma);

    // src H x W x 3 
    uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
    uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

    if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
    {
        return make_float3(-1,-1,-1);
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

__device__ 
inline float2 gaussian_interpolate_backward(uint8_t* src, float2 uv, int height, int width, float sigma, float max_dis, float3 grid_in)
{

    float item = -1.0f / (sigma * sigma);

    // src H x W x 3 
    uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
    uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

    if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
    {
        return make_float2(0,0);
    }

    int x0 = (int)uv.x;
    int y0 = (int)uv.y;

    int M = (int)(max_dis * 2) + 2; 
    int S = M / 2;

    float total_weight = 0;
    float3 dc_du = make_float3(0,0,0);
    float3 dc_dv = make_float3(0,0,0);
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

            float3 dc_ddis = c * w * item;
            dc_du += dc_ddis * (uv.x - x) * (width-1.0f);
            dc_dv += dc_ddis * (uv.y - y) * (height-1.0f);

            total_weight += w;
        }
    }
    if (total_weight > 0)
    {
        dc_du /= total_weight;
        dc_dv /= total_weight;
    }
    return make_float2(dot(grid_in,dc_du), dot(grid_in, dc_dv));
}

__global__ void gaussian_grid_sample_forward_kernel(
    uint8_t* src, // N x H x W x 3 
    float2* grid, // N x B x 1 x 2
    float3* out, // N x B x 1 x 3
    bool* mask, // N x B x 1 x 1
    int num_imgaes,
    int batch_size, 
    float sigma, float max_dis,
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int loc = blockIdx.y * batch_size + taskIdx;

        float2 uv = grid[loc];
        
        float3 color = gaussian_interpolate_forward(src + blockIdx.y * (height * width * 3), 
                                                    uv, height, width, sigma, max_dis);

        if (color.x == -1)
        {
            mask[loc] = false;
            color = make_float3(0,0,0);
        }
        else mask[loc] = true;

        out[loc] = color;

        taskIdx += total_thread;
    }
}

void gaussian_grid_sample_forward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor out,
    at::Tensor mask,
    float sigma, float max_dis)
{
    int num_imgaes = src.size(0);
    int batch_size = grid.size(1);
    int height = src.size(1);
    int width = src.size(2);

    dim3 num_block = dim3(NUM_BLOCK(batch_size), num_imgaes);
    gaussian_grid_sample_forward_kernel<<<num_block, NUM_THREAD>>>(
        (uint8_t*)src.contiguous().data_ptr<uint8_t>(),
        (float2*)grid.contiguous().data_ptr<float>(),
        (float3*)out.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        num_imgaes, batch_size, sigma, max_dis, height, width);
    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__global__ void gaussian_grid_sample_backward_kernel(
    uint8_t* src, // N x H x W x 3 
    float2* grid, // N x B x 1 x 2
    float3 * grad_in, //  N x B x 1 x 3
    float2* grad_grid, // N x B x 1 x 2
    int num_imgaes,
    int batch_size, 
    float sigma, float max_dis,
    int height, int width){

    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int loc = blockIdx.y * batch_size + taskIdx;

        float2 uv = grid[loc];
        
        grad_grid[loc] = gaussian_interpolate_backward(src + blockIdx.y * (height * width * 3), 
                                                       uv, height, width, sigma, max_dis, grad_in[loc]);

        taskIdx += total_thread;
    }
}

void gaussian_grid_sample_backward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor grad_in,
    at::Tensor grad_grid,
    float sigma, float max_dis)
{
    int num_imgaes = src.size(0);
    int batch_size = grid.size(1);
    int height = src.size(1);
    int width = src.size(2);

    dim3 num_block = dim3(NUM_BLOCK(batch_size), num_imgaes);
    gaussian_grid_sample_backward_kernel<<<num_block, NUM_THREAD>>>(
        (uint8_t*)src.contiguous().data_ptr<uint8_t>(),
        (float2*)grid.contiguous().data_ptr<float>(),
        (float3*)grad_in.contiguous().data_ptr<float>(),
        (float2*)grad_grid.contiguous().data_ptr<float>(),
        num_imgaes, batch_size, sigma, max_dis, height, width);
    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}



__global__ void grid_sample_bool_kernel(
    bool* src,
    float2* grid, // N x B x 1 x 2
    bool* out, // N x B x 1 x 1
    int num_imgaes,
    int batch_size, 
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        int loc = blockIdx.y * batch_size + taskIdx;

        float2 uv = grid[loc];
        uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
        uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);
        int x = (int)(uv.x + 0.5f);
        int y = (int)(uv.y + 0.5f);
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            bool* cur_src = src + blockIdx.y * height * width;
            out[loc] = cur_src[y * width + x];
        }
    
        taskIdx += total_thread;
    }
}

void grid_sample_bool_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor out)
{
    int num_imgaes = src.size(0);
    int batch_size = grid.size(1);
    int height = src.size(1);
    int width = src.size(2);
    dim3 num_block = dim3(NUM_BLOCK(batch_size), num_imgaes);
    grid_sample_bool_kernel<<<num_block, NUM_THREAD>>>(
        (bool*)src.contiguous().data_ptr<bool>(),
        (float2*)grid.contiguous().data_ptr<float>(),
        (bool*)out.contiguous().data_ptr<bool>(),
        num_imgaes, batch_size, height, width);
    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}



__device__ 
inline float fetch_feature(float* src, float2 uv, int height, int width)
{
    // src H x W x 3 
    // uv.x = (uv.x + 1.0f) / 2.0f * (width - 1);
    // uv.y = (uv.y + 1.0f) / 2.0f * (height - 1);

    uv.x = uv.x * width;
    uv.y = uv.y * height;
    
    // if (uv.x < 0 || uv.x >= width-1 || uv.y < 0 || uv.y >= height-1)
    // {
    //     return;
    // }
    if (uv.x < 0 || uv.x > width || uv.y < 0 || uv.y > height)
    {
        return;
    }

    int x0 = (int)uv.x;
    int y0 = (int)uv.y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float x = uv.x - x0;
    float y = uv.y - y0;

    int idx00 = x0 + y0 * width;
    int idx01 = idx00 + width;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;

    return src[idx00] * (1.0f - x) * (1.0f - y) + 
            src[idx01] * (1.0f - x) * y + 
            src[idx10] * x * (1.0f - y) + 
            src[idx11] * x * y;

}

__global__ 
void grid_sample_feature_kernel(
    float* features, // N x C x H x W
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

        float2 uv = uvs[bidx];
        int idx = view_idxs[bidx];

        if (idx == -1)
        {
            taskIdx += total_thread;
            continue;   
        }

        float* cur_feature = features + (idx * num_channel + channel) * height * width;

        output[taskIdx] = fetch_feature(cur_feature, uv, height, width);

        taskIdx += total_thread;
    }
}

void grid_sample_feature(
    at::Tensor feature, at::Tensor uvs,
    at::Tensor &output, at::Tensor view_idx)
{
    int batch_size = uvs.size(0);
    int height = feature.size(2);
    int width = feature.size(3);
    int num_channel = feature.size(1);

    grid_sample_feature_kernel<<<NUM_BLOCK(batch_size * num_channel), NUM_THREAD>>>(
        (float*)feature.contiguous().data_ptr<float>(),
        (float2*)uvs.contiguous().data_ptr<float>(),
        (float*)output.contiguous().data_ptr<float>(),
        (int*)view_idx.contiguous().data_ptr<int>(),
        num_channel, height, width, batch_size);


    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}






// __global__ 
// void sample_depth_forward(
//     float2* uvs, // B x 2
//     int* view_idx, // B x 2 
//     float* gaussian_depths, // num_views x H x W x 1
//     float* out_mu, // B x 1 
//     int height, int width, 
//     int batch_size)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 

//     while(taskIdx < batch_size)
//     {
//         float2 uv = uvs[taskIdx]; // [-1, 1]
//         int vidx = view_idx[taskIdx];

//         float* current_depth = gaussian_depths + vidx * height * width;

//         float u = 

        

//         taskIdx += total_thread;
//     }

// }