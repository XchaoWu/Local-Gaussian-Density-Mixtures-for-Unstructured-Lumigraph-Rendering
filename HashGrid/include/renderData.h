#ifndef RENDERDATA_H__
#define RENDERDATA_H__



__host__ 
void ray_overlapblock_tracing_cuda(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor &tracing_blockIdxs, // B x maxtrcing x 4 
    at::Tensor &corner, 
    float block_size,
    uint32_t log2dim);

#endif