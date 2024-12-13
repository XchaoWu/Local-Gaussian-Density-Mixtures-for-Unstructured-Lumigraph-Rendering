#ifndef ADAM_H___
#define ADAM_H___

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


void adam_step_cuda(
    at::Tensor &params, 
    at::Tensor grad_params, 
    at::Tensor &exp_avg,
    at::Tensor &exp_avg_sq,
    float lr, float beta1, float beta2,
    float eps, int &step);

void adam_step_cuda_fp16(
    at::Tensor &params, 
    at::Tensor grad_params, 
    at::Tensor &exp_avg,
    at::Tensor &exp_avg_sq,
    float lr, float beta1, float beta2,
    float eps, int &step);
#endif 