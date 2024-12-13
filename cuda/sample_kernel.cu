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


__global__ 
void background_sampling_kernel(
    float3* rays_o, 
    float3* rays_d,
    float* starts,
    float* bg_depth,
    float* z_vals,
    int num_sample,
    float sample_range,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 dir = rays_d[taskIdx];
        float t_start = starts[taskIdx];
        float depth = bg_depth[taskIdx];

        float near = fmaxf( t_start + 0.00001f,  depth - sample_range * 0.5f);
        float far = near + sample_range;

        uniform_sample_bound(z_vals+taskIdx*num_sample, near, far, num_sample);

        taskIdx += total_thread;
    }   
}



void background_sampling_cuda(
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor starts,
    at::Tensor bg_depth,
    at::Tensor &z_vals,
    int num_sample,
    float sample_range)
{
    int batch_size = rays_o.size(0);
    background_sampling_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        starts.contiguous().data_ptr<float>(),
        bg_depth.contiguous().data_ptr<float>(),
        z_vals.contiguous().data_ptr<float>(), 
        num_sample, sample_range, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}

__global__ 
void sample_insideout_block_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample,
    int num_sample_bg,
    float3* block_center,
    float3* block_size,
    int total_rays, float far,
    float* z_vals,
    float* z_vals_bg)
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    while(pixelIdx < total_rays)
    {
        float3 origin = rays_o[pixelIdx];
        float3 dir = rays_d[pixelIdx];

        float2 bound = RayAABBIntersection(origin, dir, block_center[0], block_size[0]/2.0f);

        assert(bound.x != -1.0f && bound.y != -1.0f);

        // printf("bound %f %f\n", bound.x, bound.y);

        uniform_sample_bound(z_vals + pixelIdx * num_sample, bound.x, bound.y, num_sample);
        inverse_z_sample_bound(z_vals_bg + pixelIdx * num_sample_bg, bound.y, far, num_sample_bg);

        pixelIdx += total_thread;
    }
}

void sample_insideout_block(
    at::Tensor rays_o, 
    at::Tensor rays_d,
    int num_sample, int num_sample_bg,
    at::Tensor block_center, 
    at::Tensor block_size,
    float far,
    at::Tensor &z_vals,
    at::Tensor &z_vals_bg)
{
    int num_rays = rays_o.size(0);

    sample_insideout_block_kernel<<<NUM_BLOCK(num_rays), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        num_sample, num_sample_bg, 
        (float3*)block_center.contiguous().data_ptr<float>(),
        (float3*)block_size.contiguous().data_ptr<float>(),
        num_rays, far,
        z_vals.contiguous().data_ptr<float>(),
        z_vals_bg.contiguous().data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}