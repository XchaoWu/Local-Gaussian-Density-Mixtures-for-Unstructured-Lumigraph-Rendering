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
#include "cuda_utils.h"
#include "dda.h"
#include "block_data.h"

static inline __device__ bool isOn(uint32_t n, uint64_t *mWords) 
{ 
    return 0 != (mWords[n >> 6] & (uint64_t(1) << (n & 63))); 
}

__device__ inline  
void sample_points_single_ray(
    float3 origin,
    float3 direction,
    int num_sample, 
    float* z_vals,
    uint64_t* bitmask, 
    float3 block_corner,
    float3 block_size, 
    uint3 log2dim)
{
    // printf("block_corner %f %f %f block_size %f %f %f log2dim %d %d %d \n", 
    //     block_corner.x, block_corner.y, block_corner.z,
    //     block_size.x, block_size.y, block_size.z, 
    //     log2dim.x, log2dim.y, log2dim.z);
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    if (bound.x == -1) return;

    int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;

    while(!dda.terminate())
    {
        dda.next();

        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        // uint32_t n = dda.current_tile.x * resolution.x * resolution.x + dda.current_tile.y * resolution.y + dda.current_tile.z;

        if (isOn(n, bitmask))
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
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        // uint32_t n = dda.current_tile.x * resolution.x * resolution.x + dda.current_tile.y * resolution.y + dda.current_tile.z;
        if (isOn(n, bitmask))
        {
            // printf("loc %d %d %d\n", dda.current_tile.x, dda.current_tile.y, dda.current_tile.z);
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
    DataBuffer buffer,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {

        sample_points_single_ray(rays_o[taskIdx], rays_d[taskIdx], num_sample, z_vals+taskIdx*num_sample,
                                buffer.getBitmask(), buffer.getBlockCorner(),
                                buffer.getBlockSize(), buffer.getLog2Dim());

        taskIdx += total_thread;
    }
}

void sample_points_cuda(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    int num_sample, 
    at::Tensor &z_vals,
    DataBuffer buffer)
{
    int batch_size = rays_o.size(0);


    sample_points_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        num_sample, z_vals.contiguous().data_ptr<float>(),
        buffer, batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return;
}