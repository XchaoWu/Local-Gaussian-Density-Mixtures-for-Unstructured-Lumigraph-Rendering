#include "macros.h"
#include "cutil_math.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>
#include "camera.h"

__global__ 
void compute_relate_matrix_kernel(
    float* related_matrix, // num_camera x num_tile
    PinholeCameraManager cameras, 
    float* depths, // num_camera x height x width 
    float* block_corner, float block_size,
    float tile_size, int tile_dim, int* IndexMap, 
    int height, int width, int num_camera, int num_tile)
{
    // block dim  num_camera (y)  x num_pixel (x)
    // thread dim  512 

    int num_pixel = height * width;

    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    PinholeCamera cam = cameras[blockIdx.y];

    while(taskIdx < num_pixel)
    {
        int x = taskIdx % width;
        int y = taskIdx / width;

        float d = depths[blockIdx.y * num_pixel + taskIdx];

        if (d > 0)
        {
            float3 pts = cam.pixel2world(make_float2(x,y), d);

            float3 offset = pts - make_float3(block_corner[0],block_corner[1],block_corner[2]);

            if (offset.x >= 0 && offset.x < block_size && 
                offset.y >= 0 && offset.y < block_size &&
                offset.z >= 0 && offset.z < block_size)
            {
                int3 loc = make_int3(offset / tile_size);
                int n = (loc.x << (2*tile_dim)) | (loc.y << tile_dim) | loc.z;
                int tileIdx = IndexMap[n];
                if (tileIdx != -1)
                {
                    // if(tileIdx == 8 && blockIdx.y == 21)
                    // {
                    //     printf("camera %d x %d y %d depth %f loc %d %d %d pts %f %f %f block_corner %f %f %f\n", 
                    //             blockIdx.y, x, y, d, loc.x, loc.y, loc.z, pts.x, pts.y, pts.z,
                    //             block_corner[0],block_corner[1],block_corner[2]);
                    //     // cam.K.print_data();
                    //     // cam.K.inverse().print_data();
                    // }
                    atomicAdd(related_matrix+blockIdx.y*num_tile+tileIdx, 1.0f / num_pixel);
                }
            }

        }
        taskIdx += total_thread;
    }
}

void compute_relate_matrix(
    at::Tensor &related_matrix, // num_camera x num_tile
    PinholeCameraManager cameras, 
    at::Tensor depths,
    float* block_corner, float block_size, 
    float tile_size, int tile_dim, int* IndexMap, 
    int height, int width)
{
    int num_camera = related_matrix.size(0);
    int num_tile = related_matrix.size(1);
    int num_pixel = height * width;

    compute_relate_matrix_kernel<<<dim3(NUM_BLOCK(num_pixel), num_camera), NUM_THREAD>>>(
        related_matrix.contiguous().data_ptr<float>(), cameras,
        depths.contiguous().data_ptr<float>(), block_corner, block_size, tile_size,
        tile_dim, IndexMap, height, width, num_camera, num_tile);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}