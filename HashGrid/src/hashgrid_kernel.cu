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
uint32_t hash(int3 loc, int hashmap_size)
{
    constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };
    uint32_t result = 0;
    result ^= loc.x * primes[0];
    result ^= loc.y * primes[1];
    result ^= loc.z * primes[2];

    return (hashmap_size - 1) & result;
}

__device__ inline  
void linear_weight(float* weight, float3 offset)
{
    weight[0] = (1-offset.x) * (1-offset.y) * (1-offset.z);
    weight[1] = (1-offset.x) * (1-offset.y) * offset.z;
    weight[2] = (1-offset.x) * offset.y * (1-offset.z);
    weight[3] = (1-offset.x) * offset.y * offset.z;
    weight[4] = offset.x * (1-offset.y) * (1-offset.z);
    weight[5] = offset.x * (1-offset.y) * offset.z;
    weight[6] = offset.x * offset.y * (1-offset.z);
    weight[7] = offset.x * offset.y * offset.z;
}

__device__ inline
void cal_dwdx(float* dw_dx, float3 offset)
{
    dw_dx[0] = (-1.0f) * (1-offset.y) * (1-offset.z);
    dw_dx[1] = (-1.0f) * (1-offset.y) * offset.z;
    dw_dx[2] = (-1.0f) * offset.y * (1-offset.z);
    dw_dx[3] = (-1.0f) * offset.y * offset.z;
    dw_dx[4] = (1-offset.y) * (1-offset.z);
    dw_dx[5] = (1-offset.y) * offset.z;
    dw_dx[6] = offset.y * (1-offset.z);
    dw_dx[7] = offset.y * offset.z;
}

__device__ inline  
void cal_dwdy(float* dw_dy, float3 offset)
{
    dw_dy[0] = (1-offset.x) * (-1.0f) * (1-offset.z);
    dw_dy[1] = (1-offset.x) * (-1.0f) * offset.z;
    dw_dy[2] = (1-offset.x) * (1-offset.z);
    dw_dy[3] = (1-offset.x) * offset.z;
    dw_dy[4] = offset.x * (-1.0f) * (1-offset.z);
    dw_dy[5] = offset.x * (-1.0f) * offset.z;
    dw_dy[6] = offset.x * (1-offset.z);
    dw_dy[7] = offset.x * offset.z;
}

__device__ inline  
void cal_dwdz(float* dw_dz, float3 offset)
{
    dw_dz[0] = (1-offset.x) * (1-offset.y) * (-1.0f);
    dw_dz[1] = (1-offset.x) * (1-offset.y);
    dw_dz[2] = (1-offset.x) * offset.y * (-1.0f);
    dw_dz[3] = (1-offset.x) * offset.y;
    dw_dz[4] = offset.x * (1-offset.y) * (-1.0f);
    dw_dz[5] = offset.x * (1-offset.y);
    dw_dz[6] = offset.x * offset.y * (-1.0f);
    dw_dz[7] = offset.x * offset.y;
}

__device__ inline 
void hash_feature_index(uint32_t* indices, int3 bottom_left_idx, int hashmap_size)
{
    indices[0] = hash(bottom_left_idx + make_int3(0,0,0), hashmap_size);
    indices[1] = hash(bottom_left_idx + make_int3(0,0,1), hashmap_size);
    indices[2] = hash(bottom_left_idx + make_int3(0,1,0), hashmap_size);
    indices[3] = hash(bottom_left_idx + make_int3(0,1,1), hashmap_size);
    indices[4] = hash(bottom_left_idx + make_int3(1,0,0), hashmap_size);
    indices[5] = hash(bottom_left_idx + make_int3(1,0,1), hashmap_size);
    indices[6] = hash(bottom_left_idx + make_int3(1,1,0), hashmap_size);
    indices[7] = hash(bottom_left_idx + make_int3(1,1,1), hashmap_size);
}

__device__ inline 
void hash_features(float2* hashed_features, float2* level_feature, int3 bottom_left_idx, int hashmap_size)
{
    uint32_t indices[8];
    hash_feature_index(indices, bottom_left_idx, hashmap_size);
    #pragma unroll
    for (int i=0; i<8; i++)
    {
        hashed_features[i] = level_feature[indices[i]];
    }

}


__global__ 
void embedding_forward_kernel(
    float3* points, // B x 3 
    float2* outputs, // B x (Levels x 2)
    float2* features, // Levels x 2**hashmap_size x 2
    int n_levels, int hashmap_size,
    float* block_corner,
    float* block_size, 
    int3* resolutions, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    float2* level_feature = features + blockIdx.y * hashmap_size;
    int3 level_resolution = resolutions[blockIdx.y];
    float3 corner = make_float3(block_corner[0],block_corner[1],block_corner[2]);
    float3 size = make_float3(block_size[0],block_size[1],block_size[2]);

    while(taskIdx < batch_size)
    {
        float3 pts = points[taskIdx];

        // 这里默认点必须在block内部
        pts = clamp(pts, corner, corner+size);

        // feature 存储在顶点
        float3 grid_size = size / make_float3(level_resolution-1);

        // 左下角的坐标 
        int3 bottom_left_idx =  make_int3((pts-corner)/grid_size);

        // 世界坐标系，用于计算插值的 weight 
        float3 voxel_min_vertex = make_float3(bottom_left_idx) * grid_size + corner;

        // 插值
        float3 offset = (pts - voxel_min_vertex) / grid_size;
        // printf("%f %f %f\n", offset.x, offset.y, offset.z);
        float weight[8];
        linear_weight(weight, offset);
        float2 neighbor_features[8];
        hash_features(neighbor_features, level_feature, bottom_left_idx, hashmap_size);

        float2 interpolated_feature = make_float2(0,0);
        #pragma unroll
        for (int i=0; i<8; i++)
        {
            interpolated_feature = interpolated_feature + weight[i] * neighbor_features[i];
        }
        outputs[taskIdx * n_levels + blockIdx.y] = interpolated_feature;
        
        taskIdx += total_thread;
    }
}

__global__ void embedding_backward_kernel(    
    float3* points, // B x 3 
    float2* grad_in, // B x (Levels x 2) dL_dfeature_out
    float3* grad_points, // dL_dpts  B x 3 
    float2* grad_features, // Levels x hashmap_size x 2
    float2* features, // Levels x hashmap_size x 2
    int n_levels, int hashmap_size,
    float* block_corner,
    float* block_size, 
    int3* resolutions, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    float2* level_feature = features + blockIdx.y * hashmap_size;
    int3 level_resolution = resolutions[blockIdx.y];
    float2* level_grad_features = grad_features + blockIdx.y * hashmap_size;
    float3 corner = make_float3(block_corner[0],block_corner[1],block_corner[2]);
    float3 size = make_float3(block_size[0],block_size[1],block_size[2]);

    while(taskIdx < batch_size)
    {

        float3 pts = points[taskIdx];
        float2 dL_dout = grad_in[taskIdx * n_levels + blockIdx.y];

        // 这里默认点必须在block内部
        pts = clamp(pts, corner, corner+size);

        // feature 存储在顶点
        float3 grid_size = size / make_float3(level_resolution-1);

        // 左下角的坐标 
        int3 bottom_left_idx =  make_int3((pts-corner)/grid_size);

        // 世界坐标系，用于计算插值的 weight 
        float3 voxel_min_vertex = make_float3(bottom_left_idx) * grid_size + corner;

        // 插值
        float3 offset = (pts - voxel_min_vertex) / grid_size;
        float weight[8];
        linear_weight(weight, offset);
        uint32_t indices[8];
        hash_feature_index(indices, bottom_left_idx, hashmap_size);
        float2 neighbor_features[8];
        #pragma unroll
        for (int i=0; i<8; i++)
        {
            neighbor_features[i] = level_feature[indices[i]];
        }

        #pragma unroll
        for (int i=0; i<8; i++)
        {
            // dL_dfeature 
            atomicAdd(&level_grad_features[indices[i]].x, weight[i] * dL_dout.x);
            atomicAdd(&level_grad_features[indices[i]].y, weight[i] * dL_dout.y);
        }

        float dw_dx[8], dw_dy[8], dw_dz[8];
        cal_dwdx(dw_dx, offset);
        cal_dwdy(dw_dy, offset);
        cal_dwdz(dw_dz, offset);

        float2 dout_dx=make_float2(0,0);
        float2 dout_dy=make_float2(0,0); 
        float2 dout_dz=make_float2(0,0);

        #pragma unroll
        for (int i=0; i<8; i++)
        {
            dout_dx = dout_dx + neighbor_features[i] * dw_dx[i];
            dout_dy = dout_dy + neighbor_features[i] * dw_dy[i];
            dout_dz = dout_dz + neighbor_features[i] * dw_dz[i];
        }

        atomicAdd(&grad_points[taskIdx].x, 1.0f / grid_size.x * dot(dL_dout, dout_dx));
        atomicAdd(&grad_points[taskIdx].y, 1.0f / grid_size.y * dot(dL_dout, dout_dy));
        atomicAdd(&grad_points[taskIdx].z, 1.0f / grid_size.z * dot(dL_dout, dout_dz));

        taskIdx += total_thread;
    }
}


void embedding_forward_cuda(
    at::Tensor points,
    at::Tensor &outputs,
    at::Tensor features,
    at::Tensor block_corner, 
    at::Tensor block_size,
    at::Tensor resolutions)
{
    int batch_size = points.size(0);
    int n_levels = features.size(0);
    int hashmap_size = features.size(1);
    // printf("batch_size %d n_levels %d hashmap_size %d\n", batch_size, n_levels, hashmap_size);

    embedding_forward_kernel<<<dim3(NUM_BLOCK(batch_size), n_levels), NUM_THREAD>>>(
        (float3*)points.contiguous().data_ptr<float>(),
        (float2*)outputs.contiguous().data_ptr<float>(),
        (float2*)features.contiguous().data_ptr<float>(),
        n_levels, hashmap_size, 
        (float*)block_corner.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(), 
        (int3*)resolutions.contiguous().data_ptr<int>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}

void embedding_backward_cuda(
    at::Tensor points,
    at::Tensor grad_in, 
    at::Tensor &grad_points,
    at::Tensor &grad_features,
    at::Tensor features,
    at::Tensor block_corner, 
    at::Tensor block_size,
    at::Tensor resolutions)
{
    int batch_size = points.size(0);
    int n_levels = features.size(0);
    int hashmap_size = features.size(1);


    embedding_backward_kernel<<<dim3(NUM_BLOCK(batch_size), n_levels), NUM_THREAD>>>(
        (float3*)points.contiguous().data_ptr<float>(),
        (float2*)grad_in.contiguous().data_ptr<float>(),
        (float3*)grad_points.contiguous().data_ptr<float>(),
        (float2*)grad_features.contiguous().data_ptr<float>(),
        (float2*)features.contiguous().data_ptr<float>(),
        n_levels, hashmap_size, 
        (float*)block_corner.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(),
        (int3*)resolutions.contiguous().data_ptr<int>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}
