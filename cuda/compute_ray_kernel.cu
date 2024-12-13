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
void compute_ray_forward_kernel(
    float3* rays_o, // B 
    float3* rays_d, // B 
    float* Ks,  // NUM_CAM x 9
    float* C2Ws, // NUM_CAM x 12 
    int3* locs, // B  [view_idx, px, py]
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {

        int3 loc = locs[taskIdx];

        int view_idx = loc.x;
        int x = loc.y;
        int y = loc.z;

        get_rays(x, y, Ks + view_idx*9, C2Ws + view_idx*12, rays_o[taskIdx], rays_d[taskIdx]);

        taskIdx += total_thread;
    }

}

__global__
void compute_ray_backward_kernel(
    float3* grad_rays_o,
    float3* grad_rays_d,
    float* Ks,
    float* grad_C2Ws,
    int3* locs, // B  [view_idx, px, py]
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {

        int3 loc = locs[taskIdx];

        int view_idx = loc.x;
        int _x = loc.y;
        int _y = loc.z;

        float* K = Ks + view_idx * 9;
        float x = (1.0f * _x + 0.5f - K[2]) / K[0];
        float y = (1.0f * _y + 0.5f - K[5]) / K[4];

        float* grad_c2w = grad_C2Ws + view_idx * 12;
        float3 grad_origin = grad_rays_o[view_idx];
        float3 grad_direction = grad_rays_d[view_idx];

        atomicAdd(grad_c2w+3, grad_origin.x);
        atomicAdd(grad_c2w+7, grad_origin.y);
        atomicAdd(grad_c2w+11, grad_origin.z);

        atomicAdd(grad_c2w+0, grad_direction.x * x);
        atomicAdd(grad_c2w+1, grad_direction.x * y);
        atomicAdd(grad_c2w+2, grad_direction.x);

        atomicAdd(grad_c2w+4, grad_direction.y * x);
        atomicAdd(grad_c2w+5, grad_direction.y * y);
        atomicAdd(grad_c2w+6, grad_direction.y);

        atomicAdd(grad_c2w+8, grad_direction.z * x);
        atomicAdd(grad_c2w+9, grad_direction.z * y);
        atomicAdd(grad_c2w+10, grad_direction.z);

        taskIdx += total_thread;
    }
}


void compute_ray_forward(
    at::Tensor &rays_o,
    at::Tensor &rays_d,
    at::Tensor Ks,
    at::Tensor C2Ws,
    at::Tensor locs)
{
    int batch_size = rays_o.size(0);

    compute_ray_forward_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        Ks.contiguous().data_ptr<float>(),
        C2Ws.contiguous().data_ptr<float>(),
        (int3*)locs.contiguous().data_ptr<int>(),
        batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  

}

void compute_ray_backward(
    at::Tensor grad_rays_o,
    at::Tensor grad_rays_d,
    at::Tensor Ks,
    at::Tensor &grad_C2Ws,
    at::Tensor locs)
{
    int batch_size = grad_rays_o.size(0);

    compute_ray_backward_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)grad_rays_o.contiguous().data_ptr<float>(),
        (float3*)grad_rays_d.contiguous().data_ptr<float>(),
        Ks.contiguous().data_ptr<float>(),
        grad_C2Ws.contiguous().data_ptr<float>(),
        (int3*)locs.contiguous().data_ptr<int>(),
        batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}