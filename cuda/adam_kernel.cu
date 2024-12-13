#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include "camera.h"
#include <ATen/ATen.h>
#include <stdint.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>
#include <stdio.h>
#include <math.h>              
#include <torch/extension.h>
#include "cuda_fp16.h"

#define LOSS_SCALE 128
/*
gridDim.x -> K 
gridDim.y -> dim 
*/

__global__ 
void adam_step_kernel(
    float* params, // K x 8 
    float* grad_params, // K x 8 
    float* exp_avg, 
    float* exp_avg_sq, 
    float lr,
    float beta1,
    float beta2, 
    float eps,
    int step,
    uint64_t param_size)
{
    uint32_t taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_thread = blockDim.x * gridDim.x;

    uint32_t dim = blockIdx.y;

    while(taskIdx < param_size)
    {
        uint32_t idx = taskIdx * 8 + dim;
        float grad = grad_params[idx];

        if (grad == 0.0f) 
        {
            // skip
            taskIdx += total_thread;
            continue; 
        }

        float bias_correction1 = 1.0f - powf(beta1, (float)step);
        float bias_correction2 = 1.0f - powf(beta2, (float)step);

        float exp_avg_item = beta1 * exp_avg[idx] + (1.0f - beta1) * grad;
        float exp_avg_sq_item = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * grad * grad;

        float denom = sqrtf(exp_avg_sq_item / bias_correction2) + eps;

        float step_size = lr / bias_correction1;

        params[idx] = params[idx] - step_size * exp_avg_item / denom;
        exp_avg[idx] = exp_avg_item;
        exp_avg_sq[idx] = exp_avg_sq_item;

        taskIdx += total_thread;
    }
}


void adam_step_cuda(
    at::Tensor &params, 
    at::Tensor grad_params, 
    at::Tensor &exp_avg,
    at::Tensor &exp_avg_sq,
    float lr, float beta1, float beta2,
    float eps, int &step)
{
    uint64_t param_size = params.size(0);
    uint32_t param_dim = params.size(1);

    step = step + 1;
    adam_step_kernel<<<dim3(NUM_BLOCK(param_size), param_dim, 1), NUM_THREAD>>>(
        params.contiguous().data_ptr<float>(),
        grad_params.contiguous().data_ptr<float>(),
        exp_avg.contiguous().data_ptr<float>(),
        exp_avg_sq.contiguous().data_ptr<float>(),
        lr, beta1, beta2, eps, step, param_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}


__global__ 
void adam_step_fp16_kernel(
    float* params, // K x 8 
    float* grad_params, // K x 8 
    half* exp_avg, 
    half* exp_avg_sq, 
    float lr,
    float beta1,
    float beta2, 
    float eps,
    int step,
    uint64_t param_size)
{
    uint32_t taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_thread = blockDim.x * gridDim.x;

    uint32_t dim = blockIdx.y;

    while(taskIdx < param_size)
    {
        uint32_t idx = taskIdx * 8 + dim;
        float grad = grad_params[idx] * LOSS_SCALE;

        if (grad == 0.0f) 
        {
            // skip
            taskIdx += total_thread;
            continue; 
        }

        float bias_correction1 = 1.0f - powf(beta1, (float)step);
        float bias_correction2 = 1.0f - powf(beta2, (float)step);

        // FP16    6.10 x 1e-5  -> 6 x 1e5
        float exp_avg_item = beta1 * __half2float(exp_avg[idx]) + (1.0f - beta1) * grad;
        float exp_avg_sq_item = beta2 * __half2float(exp_avg_sq[idx]) + (1.0f - beta2) * grad * grad;

        float denom = sqrtf(exp_avg_sq_item / (bias_correction2 * LOSS_SCALE * LOSS_SCALE) ) + eps;

        float step_size = lr / bias_correction1;

        params[idx] = params[idx] - step_size * exp_avg_item / (denom * LOSS_SCALE);
        exp_avg[idx] = __float2half(exp_avg_item);
        exp_avg_sq[idx] = __float2half(exp_avg_sq_item);

        taskIdx += total_thread;
    }
}


void adam_step_cuda_fp16(
    at::Tensor &params, 
    at::Tensor grad_params, 
    at::Tensor &exp_avg,
    at::Tensor &exp_avg_sq,
    float lr, float beta1, float beta2,
    float eps, int &step)
{
    uint64_t param_size = params.size(0);
    uint32_t param_dim = params.size(1);

    step = step + 1;
    adam_step_fp16_kernel<<<dim3(NUM_BLOCK(param_size), param_dim, 1), NUM_THREAD>>>(
        params.contiguous().data_ptr<float>(),
        grad_params.contiguous().data_ptr<float>(),
        (half*)exp_avg.contiguous().data_ptr<at::Half>(),
        (half*)exp_avg_sq.contiguous().data_ptr<at::Half>(),
        lr, beta1, beta2, eps, step, param_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}