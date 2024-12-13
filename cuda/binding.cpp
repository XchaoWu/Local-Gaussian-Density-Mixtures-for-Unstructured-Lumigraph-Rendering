#include "sample.h"
#include "compute_ray.h"
#include "helper.h"
#include "view_selection.h"
#include "adam.h"
#include "grid_sample.h"
#include "build_blocks.h"
#include "voxelize.h"
#include "render.h"

PYBIND11_MODULE(CUDA_EXT, m){
    m.doc() = "pybind11 torch extension";
    m.def("sample_insideout_block", &sample_insideout_block, "");

    m.def("scatter_value_cuda", &scatter_value_cuda, "");

    m.def("get_focus_cuda", &get_focus_cuda, "");

    m.def("get_neighbor_color_forward_cuda", &get_neighbor_color_forward_cuda, "");
    m.def("get_neighbor_color_backward_cuda", &get_neighbor_color_backward_cuda, "");

    m.def("get_candidate_neighbor", &get_candidate_neighbor, "");
    m.def("pixel_level_neighbor_ranking", &pixel_level_neighbor_ranking, "");
    m.def("pixel_level_pick_up_neighbor", &pixel_level_pick_up_neighbor, "");


    m.def("compute_ray_forward", &compute_ray_forward, "");
    m.def("compute_ray_backward", &compute_ray_backward, "");
    
    m.def("ray_aabb_intersection", &ray_aabb_intersection, "");
    m.def("ray_aabb_intersection_v2", &ray_aabb_intersection_v2, "");

    m.def("warping_samples_cuda", &warping_samples_cuda, "");
    m.def("neighbor_score_cuda", &neighbor_score_cuda, "");

    m.def("gen_rays_cuda", &gen_rays_cuda, "");
    m.def("gen_image_rays_cuda", &gen_image_rays_cuda, "");

    m.def("pick_up_neighbor", &pick_up_neighbor, "");

    m.def("new_neighbor_score_cuda", &new_neighbor_score_cuda, "");

    m.def("sample_points_contract", &sample_points_contract, "");
    m.def("sample_points_grid", &sample_points_grid, "");
    m.def("proj2pixel_and_fetch_color", &proj2pixel_and_fetch_color, "");

    m.def("computeViewcost", &computeViewcost, "");

    m.def("voxelize_mesh", &voxelize_mesh, "");

    m.def("background_sampling_cuda", &background_sampling_cuda, "");


    m.def("padding_inputs_forward", &padding_inputs_forward, "");
    m.def("padding_inputs_backward", &padding_inputs_backward, "");

    m.def("update_grid_value_cuda", &update_grid_value_cuda, "");

    m.def("adam_step_cuda", &adam_step_cuda, "");

    m.def("adam_step_cuda_fp16", &adam_step_cuda_fp16, "");

    m.def("grid_sample_feature", &grid_sample_feature, "");

    m.def("padding_depth", &padding_depth, "");

    m.def("project_neighbor_forward_cuda", &project_neighbor_forward_cuda, "");
    m.def("project_neighbor_backward_cuda", &project_neighbor_backward_cuda, "");


    m.def("grid_sample_forward_cuda", &grid_sample_forward_cuda, "");

    m.def("grid_sample_backward_cuda", &grid_sample_backward_cuda, "");

    m.def("gaussian_grid_sample_forward_cuda", &gaussian_grid_sample_forward_cuda, "");

    m.def("gaussian_grid_sample_backward_cuda", &gaussian_grid_sample_backward_cuda, "");

    m.def("grid_sample_bool_cuda", &grid_sample_bool_cuda, "");

    m.def("gaussian_sample", &gaussian_sample, "");

    m.def("update_voxel_grid_cuda", &update_voxel_grid_cuda, "");
    m.def("set_all_cuda", &set_all_cuda, "");
    m.def("voxel_grid_to_mask_cuda", &voxel_grid_to_mask_cuda, "");
    m.def("split_voxel_grid_cuda", &split_voxel_grid_cuda, "");
    m.def("sparse_voxel_contracted_grid_sampling_cuda", &sparse_voxel_contracted_grid_sampling_cuda, "");


    m.def("invalid_find_guide_cuda", &invalid_find_guide_cuda, "");

    m.def("get_warping_mask_cuda", &get_warping_mask_cuda, "");

    m.def("padding_results_cuda", &padding_results_cuda, "");

    m.def("grid_sample_nearest", &grid_sample_nearest, "");

    m.def("ray_cast_cuda", &ray_cast_cuda, "");

    m.def("inference_neighbor_cuda", &inference_neighbor_cuda, "");

    m.def("project_samples_cuda", &project_samples_cuda, "");

    m.def("accumulate_cuda", &accumulate_cuda, "");

    m.def("pixel_level_neighbor_ranking_render", &pixel_level_neighbor_ranking_render, "");

    m.def("sample_points_grid_render", &sample_points_grid_render, "");
    
    m.def("get_candidate_uniform_neighbor", &get_candidate_uniform_neighbor, "");
    py::class_<BlockBuilder>(m, "BlockBuilder")
        .def(py::init<>())
        .def("load_mesh", &BlockBuilder::load_mesh)
        .def("load_camera", &BlockBuilder::load_camera)
        .def("init_blocks", &BlockBuilder::init_blocks)
        .def("view_selection_for_single_block", &BlockBuilder::view_selection_for_single_block);
}