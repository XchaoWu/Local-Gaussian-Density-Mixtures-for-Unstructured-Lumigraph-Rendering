// #include "cutil_math.h"
// #include "dda_overlap.h"
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

// __device__ __constant__ float3 scene_corner;

// __device__ 
// inline void ray_overlapblock_tracing(
//     float3 origin,
//     float3 direction,
//     int* tracing_blockIdxs, // MAX_TRACING x 4 
//     float block_size, 
//     uint32_t log2dim)
// {
//     DDAOverlap dda;

//     origin = origin - scene_corner;

//     dda.init(origin, direction, make_float2(0,0), block_size, 1u << log2dim);

//     int count = 0;
//     while(!dda.terminate())
//     {
//         dda.next();

//         int bidxs[4] = {-1, -1, -1, -1};
//         int count_blocks = dda.getBlockIdx(bidxs);

//         bool update = false;

//         if (count == 0)
//         {
//             update = true;
//         }else{
//             #pragma unroll
//             for (int i=0; i<4; i++)
//             {
//                 if (bidxs[i] != tracing_blockIdxs[4*(count-1)+i])
//                 {
//                     update = true;
//                     break;
//                 }
//             }
//         }
//         if (update)
//         {
//             memcpy(tracing_blockIdxs+4*count, bidxs, sizeof(int)*4);
//             count++;
//         }

//         dda.step();
//     }
// }

// __global__ 
// void ray_overlapblock_tracing_kernel(
//     float3* rays_o,
//     float3* rays_d,
//     int* tracing_blockIdxs,
//     float block_size, 
//     uint32_t log2dim,
//     int max_tracing,
//     int batch_size)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 

//     while (taskIdx < batch_size)
//     {
//         int* cur_tracing_blockIdxs = tracing_blockIdxs+taskIdx*max_tracing*4;
//         for (int i=0; i<max_tracing*4; i++)
//             cur_tracing_blockIdxs[i] = -1;

//         ray_overlapblock_tracing(rays_o[taskIdx], rays_d[taskIdx], 
//                     cur_tracing_blockIdxs, block_size, log2dim);

//         taskIdx += total_thread;
//     }
// }

// __host__ 
// void ray_overlapblock_tracing_cuda(
//     at::Tensor rays_o,
//     at::Tensor rays_d,
//     at::Tensor &tracing_blockIdxs, // B x maxtrcing x 4 
//     at::Tensor &corner, 
//     float block_size,
//     uint32_t log2dim)
// {
//     int batch_size = rays_d.size(0);
//     int max_tracing = tracing_blockIdxs.size(1);

//     cudaMemcpyToSymbol( scene_corner.x, corner.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);

//     ray_overlapblock_tracing_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
//         (float3*)rays_o.contiguous().data_ptr<float>(),
//         (float3*)rays_d.contiguous().data_ptr<float>(),
//         (int*)tracing_blockIdxs.contiguous().data_ptr<int>(),
//         block_size, log2dim, max_tracing, batch_size);

//     AT_CUDA_CHECK(cudaGetLastError());
//     return; 

// }

















