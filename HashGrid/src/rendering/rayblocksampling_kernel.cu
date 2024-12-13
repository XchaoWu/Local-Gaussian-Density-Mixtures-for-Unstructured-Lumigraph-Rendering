// #include "cutil_math.h"
// #include "dda.h"
// #include "macros.h"
// #include "cutil_math.h"
// #include <ATen/ATen.h>
// #include <stdint.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <torch/extension.h>
// #include <ATen/TensorAccessor.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAUtils.h>
// #include <torch/extension.h>


// /*
// point 
// [block idx] [point idx] [z_val]

// batch_size x num_sample

// each block has a bitmask 
// */

// __device__ inline 
// void sample_seg(
//     float3 origin, float3 direction, 
//     float near, float far,
//     float sample_step,

// )


// __device__ inline 
// void count_seg(
//     float3 origin, float3 direction, 
//     float2 t_start, 
//     int &count,
//     float &total_lenth,
//     float3 block_corner,
//     float block_size, 
//     bool* occupied_grid,
//     uint32_t grid_log2dim)
// {
//     DDASatateScene dda;
//     uint32_t grid_resolution = 1u << grid_log2dim;
//     float grid_size = block_size / grid_resolution;
//     dda.init(origin-block_corner, direction, t_start, grid_resolution, grid_size);

//     while(!dda.terminate())
//     {
//         dda.next();

//         int3 loc = dda.current_tile;
//         int n = (loc.x << (2*grid_log2dim) ) | (loc.y << grid_log2dim) | loc.z;
//         if (occupied_grid[n] == true)
//         {

//             float len = dda.t.y - dda.t.z;
//             if (len > 0)
//             {
//                 count++;
//                 total_lenth += len;
//             }
//         }
//         dda.step();
//     }
// }

// /*
// 每个点需要在 最多 4个 block中分别判断是否处于occupied空间，
// 只要在一个occupied空间，即认为需要采样
// */
// __device__ inline 
// void sample_points_along_ray(
//     float3 origin, float3 direction, int num_sample, 
//     bool* occupied_grids, // num_blocks x size 
//     int grid_total_num, // num of grid for each block 
//     float3* block_corners, // num_blocks x 3 
//     float block_size,
//     int* tracing_blockIdxs, // MAX_TRACING x 4 
//     short* dense2sparse_block,  
//     int max_tracing,
//     short* block_idxs, // num_sample x 4
//     int* point_idxs, // num_sample
//     float* z_vals) // num_sample
// {
//     for (int i=0; i<max_tracing; i++)
//     {
//         int* cur_blockIdxs = tracing_blockIdxs + i * 4;

//         #pragma unroll
//         for (int j=0; j<4; j++)
//         {
//             int dense_bidx = cur_blockIdxs[j];
//             if (dense_bidx == -1) break;
//             short bidx = dense2sparse_block[dense_bidx];
//             float3 block_corner = block_corners[bidx];
//             bool* occupied_grid = occupied_grids + bidx * grid_total_num;



//         }
//     }
// }