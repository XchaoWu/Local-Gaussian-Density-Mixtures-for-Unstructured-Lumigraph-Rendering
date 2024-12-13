#ifndef RENDERING_H__
#define RENDERING_H__

#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/ATen.h>
#include <cassert>
#include <cuda.h>
#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include "cutil_math.h"
#include "cuda_utils.h"
 
void rendering_cuda(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor &output_diffuse,
    at::Tensor &output_specular,
    at::Tensor &output_depth,
    float sample_step,
    at::Tensor block_corners,
    at::Tensor block_sizes,
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor features_tables,
    at::Tensor resolution,
    at::Tensor params, 
    at::Tensor scene_center,
    at::Tensor scene_size);

// void rendering_outdoor_v1_cuda(
//     at::Tensor rays_o,
//     at::Tensor rays_d,
//     at::Tensor &output_diffuse,
//     at::Tensor &output_specular,
//     at::Tensor &output_depth,
//     float sample_step,
//     at::Tensor block_corners,
//     at::Tensor block_sizes,
//     at::Tensor grid_occupied, 
//     at::Tensor grid_starts, 
//     at::Tensor grid_log2dim,
//     at::Tensor features_tables,
//     at::Tensor resolution,
//     at::Tensor params, 
//     at::Tensor scene_center,
//     at::Tensor scene_size);

__host__ 
void ray_block_intersection(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor &intersections);


__host__ 
void sample_points(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor tracing_blocks,
    at::Tensor intersections, 
    at::Tensor tracing_idx,
    at::Tensor &z_start, 
    at::Tensor &z_vals,
    at::Tensor &dists);

__host__ 
void prepare_points(at::Tensor z_vals, 
                    at::Tensor runing_mask, at::Tensor intersections, at::Tensor &block_idxs);
__host__
int sort_by_key(at::Tensor &keys_tensor, at::Tensor &values_tensor, at::Tensor &starts_tensor);


__host__ 
void pts_inference(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor z_vals, at::Tensor dists,
    at::Tensor block_idxs,
    at::Tensor features_tables, at::Tensor params,
    at::Tensor resolution, 
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor &diffuse, at::Tensor &specular, at::Tensor &alpha);
__host__ 
void accumulate_color(
    at::Tensor pts_diffuse,
    at::Tensor pts_specular,
    at::Tensor pts_alpha,
    at::Tensor &transparency,
    at::Tensor z_vals,
    at::Tensor &diffuse,
    at::Tensor &specular,
    at::Tensor &depth);

__host__ 
void ray_firsthit_block(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor tracing_blocks,
    at::Tensor intersections, 
    at::Tensor &hit_blockIdxs);

__host__ 
void inverse_z_sampling(
    at::Tensor intersections,  at::Tensor related_bidx,
    at::Tensor &z_vals,  float sample_range);
__host__ 
void bg_pts_inference(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, at::Tensor outgoing_bidxs,
    at::Tensor blend_weights,
    at::Tensor block_corners, at::Tensor block_sizes, 
    at::Tensor resolution,
    at::Tensor features_tables, at::Tensor params,
    at::Tensor &diffuse, at::Tensor &specular, at::Tensor &alpha);

__host__ 
void get_last_block(
    at::Tensor tracing_blocks,
    at::Tensor &bidxs,
    at::Tensor intersections);

// __host__ 
// void update_outgoing_bidx(
//     at::Tensor rays_o, at::Tensor rays_d,
//     at::Tensor block_corners, at::Tensor block_sizes,
//     at::Tensor tracing_blocks,
//     at::Tensor intersections,
//     at::Tensor &outgoing_bidxs,
//     at::Tensor &blend_weights);

__host__ 
void update_outgoing_bidx(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor tracing_blocks,
    at::Tensor intersections,
    at::Tensor &outgoing_bidxs,
    at::Tensor &blend_weights, float ratio, bool skip);

__host__ 
void update_outgoing_bidx_v2(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor tracing_blocks,
    at::Tensor intersections,
    at::Tensor &inside_bidxs,
    at::Tensor &blend_weights);

__host__ 
void bg_pts_inference_v2(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, at::Tensor bg_idxs,
    int step,
    at::Tensor block_corners, at::Tensor block_sizes, 
    at::Tensor resolution,
    at::Tensor features_tables, at::Tensor params,
    at::Tensor &diffuse, at::Tensor &specular, at::Tensor &alpha);

__host__ 
void process_occupied_grid(
    int bidx, int total_grid,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor & tgt_grid_occupied);
#endif 