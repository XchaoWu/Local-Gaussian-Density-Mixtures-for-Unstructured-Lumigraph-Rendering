#pragma once
#include <torch/extension.h>
#include <ATen/ATen.h>

#define CHECK_INPUT(x)                                                         \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                   \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");

#define MIN_VALUE(a,b) a < b ? a : b
#define NUM_THREAD 512
#define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)