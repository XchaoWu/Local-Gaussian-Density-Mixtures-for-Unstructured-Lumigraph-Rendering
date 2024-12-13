#ifndef __RENDER_H
#define __RENDER_H



#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


void ray_cast_cuda(
    at::Tensor rt, at::Tensor k, 
    at::Tensor &rays_o, at::Tensor &rays_d,
    at::Tensor &up_axis, 
    int height, int width);

void pixel_level_neighbor_ranking_render(
    at::Tensor candidate_neighbors, 
    at::Tensor proj_mat,
    at::Tensor nei_cam_centers,
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, 
    at::Tensor up_axis,
    at::Tensor score, int num_thread,
    int height, int width);

void project_samples_cuda(
    at::Tensor origin, 
    at::Tensor samples, at::Tensor nei_idxs,
    at::Tensor proj_mat, at::Tensor nei_centers,
    at::Tensor params, 
    at::Tensor coeffi,
    int height, int width);


void inference_neighbor_cuda(
    at::Tensor origin,
    at::Tensor samples, 
    at::Tensor src_views,
    at::Tensor src_images,
    at::Tensor nei_idxs,
    at::Tensor proj_mat, at::Tensor nei_centers,
    at::Tensor network_params,
    at::Tensor bbox_center, at::Tensor bbox_size,
    at::Tensor blend_alpha, 
    at::Tensor blend_color,
    at::Tensor mask, 
    float near, float far, bool is_half);

    

void accumulate_cuda(
    at::Tensor alpha, at::Tensor T,
    at::Tensor weight);


void sample_points_grid_render(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor &dists,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird,
    float near, float far,
    int log2dim_x, int log2dim_y, int log2dim_z, bool inv_z, bool background);
// class SparseGaussian
// {

// public:
//     uint64_t* data = nullptr;

//     __host__ __device__ SparseGaussian(){}

// };

#endif 