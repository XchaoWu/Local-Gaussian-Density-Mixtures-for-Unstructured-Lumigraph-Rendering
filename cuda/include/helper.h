#ifndef HELPER_H__
#define HELPER_H__

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


void ray_aabb_intersection(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds);

void ray_aabb_intersection_v2(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds);

    
void proj2pixel_and_fetch_color(
    at::Tensor pts,
    at::Tensor Ks, 
    at::Tensor C2Ws,
    at::Tensor RGBs,
    at::Tensor &fetched_pixels,
    at::Tensor &fetched_colors);

at::Tensor sample_points_contract(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird);


void sample_points_grid(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor &dists,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird,
    float near, float far,
    int log2dim_x, int log2dim_y, int log2dim_z);

void warping_samples_cuda(
    at::Tensor rays_o, 
    at::Tensor up_axis,
    at::Tensor samples, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor images, at::Tensor valid_mask,
    at::Tensor nei_idxs,
    at::Tensor &warped_samples, 
    at::Tensor &ray_distance,
    at::Tensor &nei_bound,
    at::Tensor block_center, at::Tensor block_size, 
    at::Tensor &nei_dir,
    at::Tensor &warped_uvs,
    at::Tensor &warped_colors,
    at::Tensor &warped_colors_blur,
    at::Tensor &warped_mask, at::Tensor &warped_valid_mask,
    at::Tensor &blend_weight, float sigma, float max_dis, 
    float near, float far, 
    bool inference, int padding);


void neighbor_score_cuda(
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor samples, 
    at::Tensor depths, float dthresh,
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &score,
    int height, int width);

void new_neighbor_score_cuda(
    at::Tensor ref_idxs,
    at::Tensor up_axis,
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor samples, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor candidate_neighbors,
    at::Tensor &score, 
    int padding,
    int height, int width);

// void new_neighbor_score_cuda(
//     at::Tensor ref_idxs,
//     at::Tensor up_axis,
//     at::Tensor rays_o, 
//     at::Tensor rays_d,
//     at::Tensor samples, 
//     at::Tensor depths, float dthresh,
//     at::Tensor rts, at::Tensor ks, 
//     at::Tensor &score,
//     int height, int width);

    
void pick_up_neighbor(
    at::Tensor nei_idx, 
    at::Tensor candidate_neighbors,
    at::Tensor nei_scores, 
    at::Tensor &out);

// void pick_up_neighbor(
//     at::Tensor nei_idx, 
//     at::Tensor &out);

void gaussian_sample(
    at::Tensor images,
    at::Tensor view_idx, 
    at::Tensor uvs, at::Tensor &out_color,
    float sigma, float max_dis);


// void update_grid_value_cuda(
//     at::Tensor &grid_value, 
//     at::Tensor log2dim, float alpha, 
//     at::Tensor pts, at::Tensor value);

void padding_inputs_forward(
    at::Tensor &inputs,
    at::Tensor mask);

void padding_inputs_backward(
    at::Tensor &inputs,
    at::Tensor mask);

void padding_depth(
    at::Tensor depths, at::Tensor valid_mask,
    at::Tensor &out, int window_size);


void gen_rays_cuda(
    at::Tensor view_idxs, at::Tensor ray_idxs,
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &rays_o, at::Tensor &rays_d, at::Tensor &up_axis,
    int height, int width);

void gen_image_rays_cuda(
    int vidx,
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &rays_o, at::Tensor &rays_d, at::Tensor &up_axis,
    int height, int width);

void project_neighbor_forward_cuda(
    at::Tensor pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor &nei_rays_o, at::Tensor &nei_rays_d,
    at::Tensor &proj_uvs, at::Tensor &mask);

void project_neighbor_backward_cuda(
    at::Tensor &grad_pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor grad_in, at::Tensor mask);


void scatter_value_cuda(
    at::Tensor error_map, at::Tensor error,
    at::Tensor nei_idxs, at::Tensor warped_uv);


void get_focus_cuda(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor mask,  
    int log2dim_x, int log2dim_y, int log2dim_z,
    at::Tensor block_corner, at::Tensor block_size,
    float near, float far);


void get_neighbor_color_forward_cuda(
    at::Tensor rays_o,
    at::Tensor pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor images, at::Tensor block_center,
    at::Tensor block_size, at::Tensor &ray_distance,
    at::Tensor &nei_bound, at::Tensor &warped_uvs,
    at::Tensor &warped_color, 
    at::Tensor &blend_weight);


void get_neighbor_color_backward_cuda(
    at::Tensor &grad_pts, 
    at::Tensor pts,
    at::Tensor nei_idxs, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor images, 
    at::Tensor grad_in);


void get_candidate_neighbor(
    at::Tensor ref_idxs, at::Tensor rays_o, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor distance);

void pixel_level_neighbor_ranking(
    at::Tensor ref_idxs,
    at::Tensor candidate_neighbors, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, 
    at::Tensor up_axis,
    at::Tensor score, 
    int step, int padding, 
    int height, int width);

void pixel_level_pick_up_neighbor(
    at::Tensor sorted_idxs, 
    at::Tensor candidate_neighbors,
    at::Tensor out);

// ========== GRID =========================//

void update_voxel_grid_cuda(
    at::Tensor pts, at::Tensor value,
    at::Tensor grid_data, 
    int log2dim_x, int log2dim_y, int log2dim_z,
    float thresh);

void voxel_grid_to_mask_cuda(
    at::Tensor grid_data, 
    int log2dim_x, int log2dim_y, int log2dim_z,
    at::Tensor mask);

void split_voxel_grid_cuda(
    at::Tensor grid_data, at::Tensor new_grid_data,
    int log2dim_x, int log2dim_y, int log2dim_z);

// void sparse_voxel_contracted_grid_sampling_cuda(
//     at::Tensor rays_o, at::Tensor rays_d, 
//     float near, float far, at::Tensor grid_data,
//     int log2dim_x, int log2dim_y, int log2dim_z,
//     at::Tensor s_vals, at::Tensor s_dists);

void sparse_voxel_contracted_grid_sampling_cuda(
    at::Tensor rays_o, at::Tensor rays_d, 
    float near, float far, at::Tensor grid_mask,
    int log2dim_x, int log2dim_y, int log2dim_z,
    at::Tensor s_vals, at::Tensor s_dists);

void set_all_cuda(
    at::Tensor grid_data,
    int log2dim_x, int log2dim_y, int log2dim_z,
    bool flag);

void update_grid_value_cuda(
    at::Tensor &grid_value, 
    int log2dim_x, int log2dim_y, int log2dim_z, float alpha, 
    at::Tensor pts, at::Tensor value);



void invalid_find_guide_cuda(
    at::Tensor valid_mask, at::Tensor valid,
    at::Tensor view_idxs, at::Tensor ray_idxs,
    at::Tensor &guide_uv, int max_range);

void get_warping_mask_cuda(
    at::Tensor ref_idxs,
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor samples, 
    at::Tensor rts, at::Tensor ks, 
    at::Tensor candidate_neighbors,
    at::Tensor mask,
    int height, int width);

void padding_results_cuda(
    at::Tensor valid,
    at::Tensor img);

void grid_sample_nearest(
    at::Tensor src, at::Tensor uv,
    at::Tensor out, at::Tensor vidx,
    bool forward);

void get_candidate_uniform_neighbor(
    at::Tensor ref_idxs,  at::Tensor c2ws,
    at::Tensor nei_camera_centers,
    at::Tensor distance);
#endif 