#ifndef COMPUTE_RAY_H__
#define COMPUTE_RAY_H__

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

void compute_ray_forward(
    at::Tensor &rays_o,
    at::Tensor &rays_d,
    at::Tensor Ks,
    at::Tensor C2Ws,
    at::Tensor locs);

void compute_ray_backward(
    at::Tensor grad_rays_o,
    at::Tensor grad_rays_d,
    at::Tensor Ks,
    at::Tensor &grad_C2Ws,
    at::Tensor locs);
    
#endif 