#ifndef SAMPLE_H__
#define SAMPLE_H__

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

void sample_insideout_block(
    at::Tensor rays_o, 
    at::Tensor rays_d,
    int num_sample, int num_sample_bg,
    at::Tensor block_center, 
    at::Tensor block_size,
    float far,
    at::Tensor &z_vals,
    at::Tensor &z_vals_bg);

void background_sampling_cuda(
    at::Tensor rays_o, 
    at::Tensor rays_d,
    at::Tensor starts,
    at::Tensor bg_depth,
    at::Tensor &z_vals,
    int num_sample,
    float sample_range);
#endif 