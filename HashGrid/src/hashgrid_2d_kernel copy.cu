#include "macros.h"
#include "cutil_math.h"
#include <ATen/ATen.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>


__device__ inline
uint32_t hash(int2 loc, int hashmap_size)
{
    constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };
    uint32_t result = 0;
    result ^= loc.x * primes[0];
    result ^= loc.y * primes[1];

    return (hashmap_size - 1) & result;
}


__device__ inline 
void linear_weight(float* weight, float2 offset)
{
    weight[0] = (1-offset.x) * (1-offset.y);
    weight[1] = (1-offset.x) * offset.y;
    weight[2] = offset.x * (1-offset.y);
    weight[3] = offset.x * offset.y;
}


__device__ inline 
void hash_feature_index(uint32_t* indices, int2 bottom_left_idx, int hashmap_size)
{
    indices[0] = hash(bottom_left_idx + make_int2(0,0), hashmap_size);
    indices[1] = hash(bottom_left_idx + make_int2(0,1), hashmap_size);
    indices[2] = hash(bottom_left_idx + make_int2(1,0), hashmap_size);
    indices[3] = hash(bottom_left_idx + make_int2(1,1), hashmap_size);
}

__device__ inline 
void hash_features(float2* hashed_features, float2* level_feature, int2 bottom_left_idx, int hashmap_size)
{
    uint32_t indices[4];
    hash_feature_index(indices, bottom_left_idx, hashmap_size);
    #pragma unroll
    for (int i=0; i<4; i++)
    {
        hashed_features[i] = level_feature[indices[i]];
    }

}


__global__ 
void encoding_forward_kernel(
    float2* points, // B x 2 
    float2* outputs, // B x (Levels x 2)
    float2* features,  // Levels x (2**hashmap_size) x 2 
    int n_levels,
    int hashmap_size,
    int2* resolutions, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    float2* level_feature = features + blockIdx.y * hashmap_size;
    int2 level_resolution = resolutions[blockIdx.y];

    while(taskIdx < batch_size)
    {
        float2 pts = points[taskIdx]; // [0,1]
        float2 voxel_min_vertex = pts * make_float2(level_resolution.x-1, level_resolution.y-1);
        int2 bottom_left_idx = make_int2(voxel_min_vertex);
        float2 offset = voxel_min_vertex - make_float2(bottom_left_idx);

        float weight[4];
        linear_weight(weight, offset);

        float2 neighbor_features[4];
        hash_features(neighbor_features, level_feature, bottom_left_idx, hashmap_size);

        float2 interpolated_feature = make_float2(0,0);
        #pragma unroll
        for (int i=0; i<4; i++)
        {
            interpolated_feature = interpolated_feature + weight[i] * neighbor_features[i];
        }
        outputs[taskIdx * n_levels + blockIdx.y] = interpolated_feature;
        taskIdx += total_thread;
    }
}

void encoding_forward_cuda(
    at::Tensor points,
    at::Tensor &outputs,
    at::Tensor features,
    at::Tensor resolutions)
{
    int batch_size = points.size(0);
    int n_levels = features.size(0);
    int hashmap_size = features.size(1);
    // printf("batch_size %d n_levels %d hashmap_size %d\n", batch_size, n_levels, hashmap_size);

    encoding_forward_kernel<<<dim3(NUM_BLOCK(batch_size), n_levels), NUM_THREAD>>>(
        (float2*)points.contiguous().data_ptr<float>(),
        (float2*)outputs.contiguous().data_ptr<float>(),
        (float2*)features.contiguous().data_ptr<float>(),
        n_levels, hashmap_size, 
        (int2*)resolutions.contiguous().data_ptr<int>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}



__global__ 
void encoding_backward_kernel(
    float2* points, // B x 2
    float2* grad_in, // B x (Levels x 2) dL_dfeature_out
    float2* grad_features, // Levels x hashmap_size x 2
    float2* features, // Levels x hashmap_size x 2
    int n_levels, int hashmap_size,
    int2* resolutions, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    float2* level_feature = features + blockIdx.y * hashmap_size;
    int2 level_resolution = resolutions[blockIdx.y];
    float2* level_grad_features = grad_features + blockIdx.y * hashmap_size;

    while(taskIdx < batch_size)
    {
        
        float2 pts = points[taskIdx]; // [0,1]
        float2 voxel_min_vertex = pts * make_float2(level_resolution.x-1, level_resolution.y-1);
        int2 bottom_left_idx = make_int2(voxel_min_vertex);
        float2 offset = voxel_min_vertex - make_float2(bottom_left_idx);

        float weight[4];
        linear_weight(weight, offset);

        uint32_t indices[4];
        hash_feature_index(indices, bottom_left_idx, hashmap_size);
        float2 neighbor_features[4];
        #pragma unroll
        for (int i=0; i<4; i++)
        {
            neighbor_features[i] = level_feature[indices[i]];
        }

        
        float2 dL_dout = grad_in[taskIdx * n_levels + blockIdx.y];

        #pragma unroll
        for (int i=0; i<4; i++)
        {
            // dL_dfeature 
            atomicAdd(&level_grad_features[indices[i]].x, weight[i] * dL_dout.x);
            atomicAdd(&level_grad_features[indices[i]].y, weight[i] * dL_dout.y);
        }

        taskIdx += total_thread;
    }
}

void encoding_backward_cuda(
    at::Tensor points,
    at::Tensor grad_in, 
    at::Tensor &grad_features,
    at::Tensor features,
    at::Tensor resolutions)
{
    int batch_size = points.size(0);
    int n_levels = features.size(0);
    int hashmap_size = features.size(1);


    encoding_backward_kernel<<<dim3(NUM_BLOCK(batch_size), n_levels), NUM_THREAD>>>(
        (float2*)points.contiguous().data_ptr<float>(),
        (float2*)grad_in.contiguous().data_ptr<float>(),
        (float2*)grad_features.contiguous().data_ptr<float>(),
        (float2*)features.contiguous().data_ptr<float>(),
        n_levels, hashmap_size, 
        (int2*)resolutions.contiguous().data_ptr<int>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}
