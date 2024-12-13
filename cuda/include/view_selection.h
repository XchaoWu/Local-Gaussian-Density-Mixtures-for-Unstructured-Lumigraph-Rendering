#ifndef VIEW_SELECTION_H__
#define VIEW_SELECTION_H__

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

void computeViewcost(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor pts, 
    at::Tensor ks, 
    at::Tensor c2ws,
    at::Tensor &costs);

#endif 