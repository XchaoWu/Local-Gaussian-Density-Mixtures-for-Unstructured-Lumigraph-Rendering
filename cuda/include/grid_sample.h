#ifndef GRID_SAMPLE_H__
#define GRID_SAMPLE_H__

void grid_sample_forward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor out,
    at::Tensor mask);

void grid_sample_backward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor grad_in,
    at::Tensor grad_grid);

void gaussian_grid_sample_forward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor out,
    at::Tensor mask,
    float sigma, float max_dis);

void gaussian_grid_sample_backward_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor grad_in,
    at::Tensor grad_grid,
    float sigma, float max_dis);


void grid_sample_bool_cuda(
    at::Tensor src,
    at::Tensor grid,
    at::Tensor out);

void grid_sample_feature(
    at::Tensor feature, at::Tensor uvs,
    at::Tensor &output, at::Tensor view_idx);
    
#endif 