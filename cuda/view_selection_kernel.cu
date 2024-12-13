#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include "camera.h"
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

#define GAMMA 0.1f

__global__ 
void computeViewcost_kernel(
    float3* pts, // B  world space point 
    float3* rays_o, // B 
    float3* rays_d, // B 
    PinholeCameraManager cams, // N  all cameras 
    // float* mask, // N x B   
    // float thresh, 
    float* costs, // N x B 
    int num_cameras, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;

    // give the world 2 camera 
    PinholeCamera cam = cams[blockIdx.y];
    float3 neighbor_origin = cam.camera_center();

    while(taskIdx < batch_size)
    {

        float3 origin = rays_o[taskIdx];
        float3 direction = normalize(rays_d[taskIdx]);
        float3 p = pts[taskIdx];

        float3 uv = cam.world2pixel(p);
        if (uv.z <= 0.001) //  a small value, the point close to cam center
        {
            costs[blockIdx.y * batch_size + taskIdx] = 1.0f;
            taskIdx += total_thread;
            continue;
        }

        float3 nei_direction = normalize(p - neighbor_origin);
        // float3 nei_direction = normalize(neighbor_origin - p);

        // [0,1]
        float angle_cost = 1.0f - dot(direction, nei_direction);

        // [0,1]
        float dis_cost = fmaxf(0.0f,  1.0f - norm(p - origin) / norm(p - neighbor_origin) );

        float cost = (1.0f - GAMMA) * angle_cost + GAMMA * dis_cost;
        costs[blockIdx.y * batch_size + taskIdx] = cost;
        // if (cost > thresh) mask[blockIdx.y * batch_size + taskIdx] = 0;

        taskIdx += total_thread;
    }
}

void computeViewcost(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor pts, 
    at::Tensor ks, 
    at::Tensor c2ws,
    at::Tensor &costs)
{

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(pts);
    CHECK_INPUT(ks);
    CHECK_INPUT(c2ws);
    CHECK_INPUT(costs);

    int batch_size = costs.size(1);
    int num_camera = costs.size(0);

    // printf("num camera %d batch_size %d\n", num_camera, batch_size);

    PinholeCameraManager cams( (Intrinsic*)ks.contiguous().data_ptr<float>(), 
                               (Extrinsic*)c2ws.contiguous().data_ptr<float>(), num_camera);
    
    computeViewcost_kernel<<< dim3(NUM_BLOCK(batch_size), num_camera, 1), NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(), 
        (float3*)rays_o.contiguous().data_ptr<float>(), 
        (float3*)rays_d.contiguous().data_ptr<float>(), cams,
        costs.contiguous().data_ptr<float>(), num_camera, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
} 
