#ifndef HASHGRID__
#define HASHGRID__

#include <stdio.h>
#include "macros.h"
#include <ATen/ATen.h>
#include <stdint.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>
#include <iostream>
#include "cutil_math.h"

void embedding_forward_cuda(
    at::Tensor points,
    at::Tensor &outputs,
    at::Tensor features,
    at::Tensor block_corner, 
    at::Tensor block_size,
    at::Tensor resolutions);

void embedding_backward_cuda(
    at::Tensor points,
    at::Tensor grad_in, 
    at::Tensor &grad_points,
    at::Tensor &grad_features,
    at::Tensor features,
    at::Tensor block_corner, 
    at::Tensor block_size,
    at::Tensor resolutions);


void embedding_bg_forward_cuda(
    at::Tensor points,
    at::Tensor &outputs,
    at::Tensor features,
    at::Tensor resolutions);


void embedding_bg_backward_cuda(
    at::Tensor points,
    at::Tensor grad_in, 
    at::Tensor &grad_points,
    at::Tensor &grad_features,
    at::Tensor features,
    at::Tensor resolutions);

void encoding_forward_cuda(
    at::Tensor points,
    at::Tensor hash_idxs, 
    at::Tensor &outputs,
    at::Tensor features, // K x Levels x (2**hashmap_size) x 2
    at::Tensor resolutions);

void encoding_backward_cuda(
    at::Tensor points,
    at::Tensor hash_idxs, 
    at::Tensor grad_in, 
    at::Tensor &grad_features,
    at::Tensor features,
    at::Tensor resolutions);


#endif 