#include "render.h"
#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include <ATen/ATen.h>
#include <stdint.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuda_fp16.h"
#include <torch/extension.h>
#include "interpolation.h"
#include "dda.h"
#include "grid.h"
#include "camera.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#define PI 3.1415926
#define HIDEN_WIDTH 32
#define PARAMSIZE 1329
#define NUM_GAUSSIAN 10
#define NUM_NEIGHBOR 8

#define INV_SQRT2 1.0f / sqrtf(2.0f)
#define INV_SQRT2PI 1.0f / sqrtf(2.0f * PI)
#define SQRT2 sqrtf(2.0f)
#define SQRT3 sqrtf(3.0f)

__device__ __constant__ float3 c_origin;


template<uint32_t SIZE>
__forceinline__ __device__  
void gaussian_act(half* x)
{
    // half sigma = __float2half(0.1f);
    #pragma unroll
    for (int i=0; i<SIZE; i++)
    {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
        x[i] = hexp( __hmul( __hmul(x[i], x[i]) , __float2half(-50.0f)) );
// #else 
        
// #endif 
    }
}


template<uint32_t SIZE>
__forceinline__ __device__  
void gaussian_act(float* x)
{
    // half sigma = __float2half(0.1f);
    #pragma unroll
    for (int i=0; i<SIZE; i++)
    {
        x[i] = expf(x[i] * x[i] * -50.0f);
    }
}

template <uint32_t INPUT_DIM, uint32_t OUT_DIM, uint32_t OFFSET>
__forceinline__ __device__ 
void Linear(int &param_index, half* params, half* layer)
{
    #pragma unroll
    for (int i=OFFSET; i<OFFSET+OUT_DIM; i++)
    {
        layer[i] = params[param_index++];
    }

    #pragma unroll
    for (int i=HIDEN_WIDTH-OFFSET; i<HIDEN_WIDTH-OFFSET+INPUT_DIM; i++)
    {
        #pragma unroll
        for (int j=OFFSET; j<OFFSET+OUT_DIM; j++)
        {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
            layer[j] = __hfma(layer[i], params[param_index++], layer[j]);
// #else 
// #endif 
        }
    }
}


template <uint32_t INPUT_DIM, uint32_t OUT_DIM, uint32_t OFFSET>
__forceinline__ __device__ 
void Linear(int &param_index, float* params, float* layer)
{
    #pragma unroll
    for (int i=OFFSET; i<OFFSET+OUT_DIM; i++)
    {
        layer[i] = params[param_index++];
    }

    #pragma unroll
    for (int i=HIDEN_WIDTH-OFFSET; i<HIDEN_WIDTH-OFFSET+INPUT_DIM; i++)
    {
        #pragma unroll
        for (int j=OFFSET; j<OFFSET+OUT_DIM; j++)
        {
            layer[j] = layer[i] * params[param_index++] + layer[j];
        }
    }
}


__global__ 
void ray_cast_kernel(
    PinholeCameraManager cams, 
    float3* rays_o, // B x 3
    float3* rays_d, // B x 3 
    float3* up_axis,
    int height, int width)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < height*width)
    {
        float2 uv = make_float2((float)(taskIdx % width) + 0.5f, 
                                (float)(taskIdx / width) + 0.5f);

        PinholeCamera cam = cams[0];
        float3 x_world = cam.pixel2world(uv, 1.0f);

        float3 origin = cam.camera_center();

        rays_o[taskIdx] = origin;
        rays_d[taskIdx] = x_world - origin;
        up_axis[taskIdx] = make_float3(cam.E.data[4], cam.E.data[5], cam.E.data[6]);

        taskIdx += total_thread;
    }
}


void ray_cast_cuda(
    at::Tensor rt, at::Tensor k, 
    at::Tensor &rays_o, at::Tensor &rays_d,
    at::Tensor &up_axis, 
    int height, int width)
{
    // PinholeCamera cam(Intrinsic(k.contiguous().data_ptr<float>()),
    //                   Extrinsic(c2w.contiguous().data_ptr<float>()));
    // PinholeCamera cam(k.contiguous().data_ptr<float>(), 
                    //   c2w.contiguous().data_ptr<float>());

    PinholeCameraManager cameras((Intrinsic*)k.contiguous().data_ptr<float>(), 
                                (Extrinsic*)rt.contiguous().data_ptr<float>(), 1);

    ray_cast_kernel<<<NUM_BLOCK(height*width), NUM_THREAD>>>(
        cameras, (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)up_axis.contiguous().data_ptr<float>(),
        height, width);


    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


__device__ 
inline float3 fetch_color(uint8_t* src, float2 uv, int height, int width,
              bool &valid)
{

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    if (uv.x < 0 || uv.x >= width || uv.y < 0 || uv.y >= height)
    {
        valid = false;
        return color;
    }

    valid = true;

    // 实现0.5,0.5像素中心版本的双线性插值 
    float total_weight = 0.0f; 
    

    uv.x = uv.x - 0.5f;
    uv.y = uv.y - 0.5f;
    int u = (int)floorf(uv.x);
    int v = (int)floorf(uv.y);

    float2 w_base = make_float2(uv.x - u, uv.y - v);
    float w[4] = {  
        (1.0f - w_base.x) * (1.0f - w_base.y),
        (1.0f - w_base.x) * w_base.y,
        w_base.x * (1.0f - w_base.y),
        w_base.x * w_base.y};

    #pragma unroll
    for(int i=0; i<2; i++)
    {
        #pragma unroll
        for(int j=0; j<2; j++)
        {
            int x = u + i;
            int y = v + j;
            if (x < 0 || y < 0 || x >= width || y >= height) continue;

            float weight = w[i*2+j];
            total_weight += weight;
            int idx = y * width + x;
            color.x += weight * (float)src[idx * 3 + 0];
            color.y += weight * (float)src[idx * 3 + 1];
            color.z += weight * (float)src[idx * 3 + 2];
        }
    }

    if (total_weight > 0.0f)
    {
        color = color / total_weight;
    }
    return color;
}

__device__ inline
float gaussian(float z, float mu, float inv_sigma)
{   
    float item = (z - mu) * inv_sigma;
    return inv_sigma * expf(-0.5f * item * item);
}


// __device__ inline
// float gaussian(half z, half mu, half inv_sigma)
// {   

//     half item = __hmul(__hsub(z, mu), inv_sigma);
//     return __half2float(__hmul(inv_sigma, hexp(__hmul(__hmul(item, item)) ,__float2half(-0.5f) )));
// }

__device__ inline 
float integrated_gaussian_func(float z, float mu, float inv_sigma, float near)
{

    float item1 = inv_sigma * INV_SQRT2;
    float item2 = item1 * mu;
    return fmaxf((erff(item1 * z - item2) - erff(item1 * near - item2)), 0.0f);
    
}



__global__ 
void project_samples_kernel(
    float3* samples, // B x num_sample x 3 
    int* nei_idxs, // B x num_neighbor x 1 
    float* proj_mat, // num_view x 3 x 4 
    float3* nei_centers, // num_view x 3 
    half* network_params, 
    // float3* blend_color, // B x num_sample  x 1
    // float* blend_alpha, //B x num_sample  x 1
    float* coeffi, // B x num_sample x num_neighbor x 1
    int height, int width,
    int num_sample,
    int num_view,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    __shared__ half network_cache[PARAMSIZE];

    int load_idx = threadIdx.x;
    while (load_idx < PARAMSIZE) {
        network_cache[load_idx] = network_params[load_idx];
        load_idx += blockDim.x;
    }
    __syncthreads();

    half layer[HIDEN_WIDTH*2];

    int total_task = batch_size*num_sample;
    while(taskIdx < total_task)
    {

        int batch_idx = taskIdx / num_sample;

        float3 x_world = samples[taskIdx];
        
        int vidx = nei_idxs[batch_idx * NUM_NEIGHBOR + blockIdx.y];


        if (vidx == -1)
        {
            taskIdx += total_thread;
            continue;
        }


        float* mat = proj_mat + vidx * 12;

        float u = mat[0] * x_world.x + mat[1] * x_world.y + mat[2] * x_world.z + mat[3];
        float v = mat[4] * x_world.x + mat[5] * x_world.y + mat[6] * x_world.z + mat[7];
        float w = mat[8] * x_world.x + mat[9] * x_world.y + mat[10] * x_world.z + mat[11];

        if (w <= 0) 
        {
            taskIdx += total_thread;
            continue;
        }


        u = u / w;
        v = v / w;

        if (u < 0 || v < 0 || u > width || v > height) 
        {
            taskIdx += total_thread;
            continue;
        }

        float3 nei_dir = normalize(x_world - c_origin) - normalize(x_world - nei_centers[vidx]);

        // inference
        layer[0] = __float2half(x_world.x);
        layer[1] = __float2half(x_world.y);
        layer[2] = __float2half(x_world.z);

        int param_index = 0;
        Linear<3, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, network_cache, layer);
        gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);

        Linear<HIDEN_WIDTH, 16, 0>(param_index, network_cache, layer);
        gaussian_act<HIDEN_WIDTH>(layer);

        Linear<HIDEN_WIDTH, 16, HIDEN_WIDTH>(param_index, network_cache, layer);

        layer[16+0] = __float2half(nei_dir.x);
        layer[16+1] = __float2half(nei_dir.y);
        layer[16+2] = __float2half(nei_dir.z);


        Linear<19, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, network_cache, layer);
        gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);

        Linear<HIDEN_WIDTH, 1, 0>(param_index, network_cache, layer);
        // assert (param_index == PARAMSIZE);

        coeffi[taskIdx*NUM_NEIGHBOR + blockIdx.y] = 1.0f / (1.0f + expf(-1.0f * __half2float(layer[0])));



        taskIdx += total_thread;

    }   
}


void project_samples_cuda(
    at::Tensor origin, 
    at::Tensor samples, at::Tensor nei_idxs,
    at::Tensor proj_mat, at::Tensor nei_centers,
    at::Tensor params, 
    at::Tensor coeffi,
    int height, int width)
{
    int batch_size = samples.size(0);
    int num_sample = samples.size(1);
    int num_view = proj_mat.size(0);


    // cudaMemcpyToSymbol( c_origin.x, origin.contiguous().data_ptr<float>(), sizeof(float)*3, 0, 
    //                   cudaMemcpyDeviceToDevice);

    int n_thread = 256;
    int n_blocks = min(65535, (batch_size*num_sample + n_thread - 1) / n_thread);
    

    project_samples_kernel<<<dim3(n_blocks, NUM_NEIGHBOR), n_thread>>>(
        (float3*)samples.contiguous().data_ptr<float>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(),
        (float*)proj_mat.contiguous().data_ptr<float>(),
        (float3*)nei_centers.contiguous().data_ptr<float>(),
        (half*)params.contiguous().data_ptr<at::Half>(),
        (float*)coeffi.contiguous().data_ptr<float>(),
        height, width,
        num_sample, num_view, batch_size);


    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


/////////////// HALF 
__global__ 
void inference_neighbor_half_kernel(
    float3* samples, // B x num_sample x 3 
    // float* coeffi, // B x num_sample x num_neighbor x 1
    half* src_views, // num_view x H x W * (3*num_gaussian)
    uint8_t* src_images, // num_view x H x W * 3
    int* nei_idxs, // B x num_neighbor x 1 
    float* proj_mat, // num_view x 3 x 4 
    float3* nei_centers, // num_view x 3 
    half* network_params,
    float3* bbox_center,
    float3* bbox_size,
    float* blend_alpha, // B x num_sample x 1
    float3* blend_color, // B x num_sample x 3
    bool* mask, // B x num_sample x 1
    float near, float far,
    int num_sample,
    int num_view,
    int height, int width,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    // extern __shared__ float proj_mat_shared[];

    // int load_idx = threadIdx.x;
    // while (load_idx < num_view*12) {
    //     proj_mat_shared[load_idx] = proj_mat[load_idx];
    //     load_idx += blockDim.x;
    // }

    __shared__ half network_cache[PARAMSIZE];

    int load_idx = threadIdx.x;
    while (load_idx < PARAMSIZE) {
        network_cache[load_idx] = network_params[load_idx];
        load_idx += blockDim.x;
    }

    __syncthreads();

    half layer[HIDEN_WIDTH*2];

    uint64_t STEP = height * width * 3 * NUM_GAUSSIAN;

    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int point_idx = taskIdx % num_sample;


        float3 x_world = samples[taskIdx];
        int* cur_nei_idxs = nei_idxs + batch_idx * NUM_NEIGHBOR;

        float point_alpha = 0.0f;
        float weight_geo = 1e-8;

        float3 point_color = make_float3(0.0f,0.0f,0.0f);
        float weight_rgb = 1e-8;


        #pragma unroll
        for (int k=0; k<NUM_NEIGHBOR; k++)
        {
            int vidx = cur_nei_idxs[k];

            if (vidx == -1) continue;

            // PinholeCamera cam = cameras[vidx];
            float* mat = proj_mat + vidx * 12;
            half* cur_src_view = src_views + (uint64_t)vidx * STEP;

            float u = mat[0] * x_world.x + mat[1] * x_world.y + mat[2] * x_world.z + mat[3];
            float v = mat[4] * x_world.x + mat[5] * x_world.y + mat[6] * x_world.z + mat[7];
            float w = mat[8] * x_world.x + mat[9] * x_world.y + mat[10] * x_world.z + mat[11];

            if (w <= 0) continue;

            u = u / w;
            v = v / w;

            
            uint8_t* cur_src_img = src_images + vidx * height * width * 3;
            bool valid;
            float3 warped_color = fetch_color(cur_src_img, make_float2(u,v), height, width, valid);


            // printf("finished fetch color  valid %d\n", valid);

            if (valid == false) continue;



            mask[taskIdx] = true;

            float3 nei_c = nei_centers[vidx];

            float3 nei_dir = normalize(x_world - c_origin) - normalize(x_world - nei_c);

            float z = norm(x_world - nei_c);

            float alpha = 0.0f;
            float visibility = 0.0f;

            half* para = cur_src_view + ((int)v * width + (int)u) * 3 * NUM_GAUSSIAN;

            // printf("start fetch Gaussian Mix height %d width %d\n", height, width);

            #pragma unroll
            for (int i=0; i<NUM_GAUSSIAN; i++)
            {
                float mu = __half2float(para[i*3+0]);
                float inv_sigma = __half2float(para[i*3+1]);
                float weight = __half2float(para[i*3+2]);

                // float mu = para[i*3+0];
                // float inv_sigma = para[i*3+1];
                // float weight = para[i*3+2];

                // if (point_idx == 0){
                // printf("vidx %d Guassian %d mu %f inv_sigma %f weight %f alpha %f\n",
                //         vidx, i, mu, inv_sigma, weight, alpha);
                // }


                alpha += weight * gaussian(z, mu, inv_sigma);

                visibility += weight * integrated_gaussian_func(z, mu, inv_sigma, near);
            }
            visibility = expf(-0.5f * visibility);


            // ======================= INFERENCE coeffi ====================================
            // x_world = (x_world - bbox_center[0]) * 2.0f  / bbox_size[0];


            float3 x_nor = (x_world - bbox_center[0]) * 2.0f  / bbox_size[0];

            layer[0] = __float2half( x_nor.x );
            layer[1] = __float2half( x_nor.y );
            layer[2] = __float2half( x_nor.z );

            int param_index = 0;
            Linear<3, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, network_cache, layer);
            gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);


            Linear<HIDEN_WIDTH, 16, 0>(param_index, network_cache, layer);

            layer[16+0] = __float2half(nei_dir.x);
            layer[16+1] = __float2half(nei_dir.y);
            layer[16+2] = __float2half(nei_dir.z);

            Linear<19, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, network_cache, layer);
            gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);

            Linear<HIDEN_WIDTH, 1, 0>(param_index, network_cache, layer);

            float coeffi = 1.0f / (1.0f + expf(-1.0f * __half2float(layer[0])));


            // float coeffi = 1.0f;
            // coeffi = coeffi * 100.0f;
            // ======================= INFERENCE coeffi ====================================

            // printf("point idx %d point (%f %f %f) vidx %d uv (%f, %f) color (%f %f %f) valid %d coeffi %f no_coeffi %d\n",
            //         point_idx, x_world.x, x_world.y, x_world.z,
            //         vidx, u,v, warped_color.x, warped_color.y, warped_color.z, valid, coeffi, no_coeffi);


            point_alpha += alpha * visibility;
            point_color += warped_color * visibility * coeffi;

            weight_geo += visibility;
            weight_rgb += coeffi * visibility;

        }


        blend_alpha[taskIdx] = INV_SQRT2PI * point_alpha / weight_geo;
        blend_color[taskIdx] = point_color / weight_rgb;

        taskIdx += total_thread;
        continue;
    }
}


// [TODO] unroll with fixed num_gaussian
__global__ 
void inference_neighbor_kernel(
    float3* samples, // B x num_sample x 3 
    // float* coeffi, // B x num_sample x num_neighbor x 1
    float* src_views, // num_view x H x W * (3*num_gaussian)
    uint8_t* src_images, // num_view x H x W * 3
    int* nei_idxs, // B x num_neighbor x 1 
    float* proj_mat, // num_view x 3 x 4 
    float3* nei_centers, // num_view x 3 
    half* network_params,
    float3* bbox_center,
    float3* bbox_size,
    float* blend_alpha, // B x num_sample x 1
    float3* blend_color, // B x num_sample x 3
    bool* mask, // B x num_sample x 1
    float near, float far,
    int num_sample,
    int num_view,
    int height, int width,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    // extern __shared__ float proj_mat_shared[];

    // int load_idx = threadIdx.x;
    // while (load_idx < num_view*12) {
    //     proj_mat_shared[load_idx] = proj_mat[load_idx];
    //     load_idx += blockDim.x;
    // }

    __shared__ half network_cache[PARAMSIZE];

    int load_idx = threadIdx.x;
    while (load_idx < PARAMSIZE) {
        network_cache[load_idx] = network_params[load_idx];
        load_idx += blockDim.x;
    }

    __syncthreads();

    half layer[HIDEN_WIDTH*2];

    uint64_t STEP = height * width * 3 * NUM_GAUSSIAN;

    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int point_idx = taskIdx % num_sample;


        float3 x_world = samples[taskIdx];
        int* cur_nei_idxs = nei_idxs + batch_idx * NUM_NEIGHBOR;

        float point_alpha = 0.0f;
        float weight_geo = 1e-8;

        float3 point_color = make_float3(0.0f,0.0f,0.0f);
        float weight_rgb = 1e-8;


        #pragma unroll
        for (int k=0; k<NUM_NEIGHBOR; k++)
        {
            int vidx = cur_nei_idxs[k];

            if (vidx == -1) continue;

            // PinholeCamera cam = cameras[vidx];
            float* mat = proj_mat + vidx * 12;
            float* cur_src_view = src_views + (uint64_t)vidx * STEP;

            float u = mat[0] * x_world.x + mat[1] * x_world.y + mat[2] * x_world.z + mat[3];
            float v = mat[4] * x_world.x + mat[5] * x_world.y + mat[6] * x_world.z + mat[7];
            float w = mat[8] * x_world.x + mat[9] * x_world.y + mat[10] * x_world.z + mat[11];

            if (w <= 0) continue;

            u = u / w;
            v = v / w;

            
            uint8_t* cur_src_img = src_images + vidx * height * width * 3;
            bool valid;
            float3 warped_color = fetch_color(cur_src_img, make_float2(u,v), height, width, valid);


            // printf("finished fetch color  valid %d\n", valid);

            if (valid == false) continue;



            mask[taskIdx] = true;

            float3 nei_c = nei_centers[vidx];

            float3 nei_dir = normalize(x_world - c_origin) - normalize(x_world - nei_c);

            float z = norm(x_world - nei_c);

            float alpha = 0.0f;
            float visibility = 0.0f;

            float* para = cur_src_view + ((int)v * width + (int)u) * 3 * NUM_GAUSSIAN;

            // printf("start fetch Gaussian Mix height %d width %d\n", height, width);

            #pragma unroll
            for (int i=0; i<NUM_GAUSSIAN; i++)
            {
                // float mu = __half2float(para[i*3+0]);
                // float inv_sigma = __half2float(para[i*3+1]);
                // float weight = __half2float(para[i*3+2]);

                float mu = para[i*3+0];
                float inv_sigma = para[i*3+1];
                float weight = para[i*3+2];

                // if (point_idx == 0){
                // printf("vidx %d Guassian %d mu %f inv_sigma %f weight %f alpha %f\n",
                //         vidx, i, mu, inv_sigma, weight, alpha);
                // }


                alpha += weight * gaussian(z, mu, inv_sigma);

                visibility += weight * integrated_gaussian_func(z, mu, inv_sigma, near);
            }
            visibility = expf(-0.5f * visibility);


            // ======================= INFERENCE coeffi ====================================
            // x_world = (x_world - bbox_center[0]) * 2.0f  / bbox_size[0];


            float3 x_nor = (x_world - bbox_center[0]) * 2.0f  / bbox_size[0];

            layer[0] = __float2half( x_nor.x );
            layer[1] = __float2half( x_nor.y );
            layer[2] = __float2half( x_nor.z );

            int param_index = 0;
            Linear<3, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, network_cache, layer);
            gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);


            Linear<HIDEN_WIDTH, 16, 0>(param_index, network_cache, layer);

            layer[16+0] = __float2half(nei_dir.x);
            layer[16+1] = __float2half(nei_dir.y);
            layer[16+2] = __float2half(nei_dir.z);

            Linear<19, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, network_cache, layer);
            gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);

            Linear<HIDEN_WIDTH, 1, 0>(param_index, network_cache, layer);

            float coeffi = 1.0f / (1.0f + expf(-1.0f * __half2float(layer[0])));

            // float coeffi = expf(-1.0 * (1.0 - dot(normalize(x_world - c_origin), normalize(x_world - nei_c))));

            // float coeffi = 1.0f;
            // coeffi = coeffi * 100.0f;
            // ======================= INFERENCE coeffi ====================================

            // printf("point idx %d point (%f %f %f) vidx %d uv (%f, %f) color (%f %f %f) valid %d coeffi %f no_coeffi %d\n",
            //         point_idx, x_world.x, x_world.y, x_world.z,
            //         vidx, u,v, warped_color.x, warped_color.y, warped_color.z, valid, coeffi, no_coeffi);


            point_alpha += alpha * visibility;
            point_color += warped_color * visibility * coeffi;

            weight_geo += visibility;
            weight_rgb += coeffi * visibility;

        }


        blend_alpha[taskIdx] = INV_SQRT2PI * point_alpha / weight_geo;
        blend_color[taskIdx] = point_color / weight_rgb;

        taskIdx += total_thread;
        continue;
    }
}


void inference_neighbor_cuda(
    at::Tensor origin,
    at::Tensor samples, 
    at::Tensor src_views,
    at::Tensor src_images,
    at::Tensor nei_idxs,
    at::Tensor proj_mat, at::Tensor nei_centers,
    at::Tensor network_params,
    at::Tensor bbox_center, at::Tensor bbox_size,
    at::Tensor blend_alpha, 
    at::Tensor blend_color,
    at::Tensor mask, 
    float near, float far, bool is_half)
{
    int batch_size = blend_alpha.size(0);
    int num_sample = blend_alpha.size(1);
    int num_view = proj_mat.size(0);
    int height = src_images.size(1);
    int width = src_images.size(2);

    // int num_gaussian = src_views.size(3) / 3;

    // printf("========== src_views shape %d %d %d %d\n", 
    // src_views.size(0), src_views.size(1), src_views.size(2), src_views.size(3));

    cudaMemcpyToSymbol( c_origin.x, origin.contiguous().data_ptr<float>(), sizeof(float)*3, 0, 
                  cudaMemcpyDeviceToDevice);


    // printf("batch_size %d num_sample %d num_neighbor %d num_view %d height %d width %d num_gaussian %d\n",
    // batch_size, num_sample, num_neighbor, num_view, height, width, num_gaussian);

    int num_thread = 256;
    int n_blocks = min(65535, (batch_size*num_sample + num_thread - 1) / num_thread);

    // cudaMemcpyToSymbol(AABB_center.x, bbox_center.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);
    // cudaMemcpyToSymbol(AABB_size.x, bbox_size.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);

    if (is_half)
    {
    inference_neighbor_half_kernel<<<n_blocks, num_thread>>>(
        (float3*)samples.contiguous().data_ptr<float>(),
        (half*)src_views.contiguous().data_ptr<at::Half>(),
        // (float*)src_views.contiguous().data_ptr<float>(),
        (uint8_t*)src_images.contiguous().data_ptr<uint8_t>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(),
        (float*)proj_mat.contiguous().data_ptr<float>(),
        (float3*)nei_centers.contiguous().data_ptr<float>(),
        (half*)network_params.contiguous().data_ptr<at::Half>(),
        // (float*)network_params.contiguous().data_ptr<float>(),
        (float3*)bbox_center.contiguous().data_ptr<float>(),
        (float3*)bbox_size.contiguous().data_ptr<float>(),
        (float*)blend_alpha.contiguous().data_ptr<float>(),
        (float3*)blend_color.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        near, far, num_sample, num_view,
        height, width, batch_size);
    }else{

    inference_neighbor_kernel<<<n_blocks, num_thread>>>(
        (float3*)samples.contiguous().data_ptr<float>(),
        // (half*)src_views.contiguous().data_ptr<at::Half>(),
        (float*)src_views.contiguous().data_ptr<float>(),
        (uint8_t*)src_images.contiguous().data_ptr<uint8_t>(),
        (int*)nei_idxs.contiguous().data_ptr<int>(),
        (float*)proj_mat.contiguous().data_ptr<float>(),
        (float3*)nei_centers.contiguous().data_ptr<float>(),
        (half*)network_params.contiguous().data_ptr<at::Half>(),
        // (float*)network_params.contiguous().data_ptr<float>(),
        (float3*)bbox_center.contiguous().data_ptr<float>(),
        (float3*)bbox_size.contiguous().data_ptr<float>(),
        (float*)blend_alpha.contiguous().data_ptr<float>(),
        (float3*)blend_color.contiguous().data_ptr<float>(),
        (bool*)mask.contiguous().data_ptr<bool>(),
        near, far, num_sample, num_view,
        height, width, batch_size);

    }


    AT_CUDA_CHECK(cudaGetLastError());
    return;

}


__global__ 
void accumulate_kernel(
    float* alpha, // B x num_sample x 1
    float* T, // B x 1
    float* weight, // B x num_sample x 1
    int num_sample,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        float* cur_alpha = alpha + taskIdx * num_sample;
        float* cur_weight = weight + taskIdx * num_sample;

        float transparency = T[taskIdx];
        for (int i=0; i<num_sample; i++)
        {
            float a = cur_alpha[i];
            cur_weight[i] = transparency * a;

            transparency = transparency * (1 - a);
        }
        T[taskIdx] = transparency;
        
        taskIdx += total_thread;
    }
}


void accumulate_cuda(
    at::Tensor alpha, at::Tensor T,
    at::Tensor weight)
{
    int batch_size = alpha.size(0);
    int num_sample = alpha.size(1);
    accumulate_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float*)alpha.contiguous().data_ptr<float>(),
        (float*)T.contiguous().data_ptr<float>(),
        (float*)weight.contiguous().data_ptr<float>(),
        num_sample, batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return;

}



__global__ 
void inference_coeffi_kernel(
    float3* samples, // B x num_sample x 3
    float3* nei_dirs, // B x num_sample x num_neighbor x 3
    float* visibility, // B x num_sample x num_neighbor x 1
    float* coeffi, // B x num_sample x num_neighbor x 1
    half* network_params, 
    int batch_size, int num_sample, int num_neighbor)
{

    __shared__ half network_cache[PARAMSIZE];

    int load_idx = threadIdx.x;
    while (load_idx < PARAMSIZE) {
        network_cache[load_idx] = network_params[load_idx];
        load_idx += blockDim.x;
    }
    __syncthreads();


    half layer[HIDEN_WIDTH*2];

    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size*num_sample)
    {
        float3 point = samples[taskIdx];
        float3 direction = nei_dirs[taskIdx * num_neighbor + blockIdx.y];
        float vis = visibility[taskIdx * num_neighbor + blockIdx.y];

        layer[0] = __float2half(point.x);
        layer[1] = __float2half(point.y);
        layer[2] = __float2half(point.z);

        int param_index = 0;
        Linear<3, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, network_cache, layer);
        gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);

        Linear<HIDEN_WIDTH, HIDEN_WIDTH, 0>(param_index, network_cache, layer);
        gaussian_act<HIDEN_WIDTH>(layer);

        Linear<HIDEN_WIDTH, 16, HIDEN_WIDTH>(param_index, network_cache, layer);

        layer[HIDEN_WIDTH+16+0] = __float2half(direction.x);
        layer[HIDEN_WIDTH+16+1] = __float2half(direction.y);
        layer[HIDEN_WIDTH+16+2] = __float2half(direction.z);
        layer[HIDEN_WIDTH+16+3] = __float2half(vis);


        Linear<20, HIDEN_WIDTH, 0>(param_index, network_cache, layer);
        gaussian_act<HIDEN_WIDTH>(layer);

        Linear<HIDEN_WIDTH, 1, HIDEN_WIDTH>(param_index, network_cache, layer);


        coeffi[taskIdx * num_neighbor + blockIdx.y] = 
                    1.0f / (1.0f + expf(-1.0f * __half2float(layer[HIDEN_WIDTH])));

        taskIdx += total_thread;
        continue;
    }
}

void inference_coeffi_cuda(
    at::Tensor samples, at::Tensor nei_dirs,
    at::Tensor visibility, at::Tensor coeffi,
    at::Tensor network_params)
{
    int batch_size = nei_dirs.size(0);
    int num_sample = nei_dirs.size(1);
    int num_neighbor = nei_dirs.size(2);
    
    inference_coeffi_kernel<<<NUM_BLOCK(batch_size*num_sample), NUM_THREAD>>>(
        (float3*)samples.contiguous().data_ptr<float>(),
        (float3*)nei_dirs.contiguous().data_ptr<float>(),
        (float*)visibility.contiguous().data_ptr<float>(),
        (float*)coeffi.contiguous().data_ptr<float>(),
        (half*)network_params.contiguous().data_ptr<at::Half>(),
        batch_size, num_sample, num_neighbor);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}

// __global__ 
// void blend_neighbor_kernel(
//     float* alpha, // B x num_sample x num_neighbor x 1
//     float* visibility, // B x num_sample x num_neighbor x 1
//     float3* warped_color, // B x num_sample x num_neighbor x 3
//     float3* blend_weight, // 
//     float* blend_alpha, // B x num_sample x 1
//     float3* blend_color, // B x num_sample x 3
//     int batch_size, int num_sample,
//     int num_neighbor)
// {
//     // inference network here  (Fully fused MLP)
// }



__global__ 
void pixel_level_neighbor_ranking_kernel(
    int* candidate_neighbors, // B x K 
    float* proj_mat, // N x 3 x 4
    float3* nei_cam_centers, // N x 3 
    float3* rays_o, // B x 3 
    float3* rays_d, // B x 3 
    float* z_vals, // B x num_sample x 1  TODO z_vals
    float3* up_axis, // B x 3
    float* score, // B x K x 4 
    int height, int width,
    int num_candidate, // K
    int num_sample,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size*num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int point_idx = taskIdx % num_sample;

        float3 origin = rays_o[batch_idx];
        float3 direction = rays_d[batch_idx];
        float z_depth = z_vals[taskIdx];
        float3 x_world = origin + z_depth * direction;
        float3 normal = normalize(origin - x_world);

        float3 up = up_axis[batch_idx];

        float3 basis_a = normalize(cross(normal, up));
        float3 basis_b = normalize(cross(normal, basis_a));
        

        float* cur_score = score + batch_idx * (num_candidate * 4);

        

        for(int k=0; k<num_candidate; k++)
        {
            int nei_idx = candidate_neighbors[batch_idx*num_candidate+k];


            float* mat = proj_mat + nei_idx * 12;
            
            float u = mat[0] * x_world.x + mat[1] * x_world.y + mat[2] * x_world.z + mat[3];
            float v = mat[4] * x_world.x + mat[5] * x_world.y + mat[6] * x_world.z + mat[7];
            float w = mat[8] * x_world.x + mat[9] * x_world.y + mat[10] * x_world.z + mat[11];

            if (w<= 0)  
            {
                cur_score = cur_score + 4;
                continue;
            }

            u = u / w;
            v = v / w;
            if (u < 0 || v < 0 || u >= width || v >= height) 
            {
                cur_score = cur_score + 4;
                continue;
            }


            float3 camera_center = nei_cam_centers[nei_idx];


            float3 view_drection = normalize(camera_center - x_world);


            float cos_sim = view_drection.x * normal.x + 
                            view_drection.y * normal.y + 
                            view_drection.z * normal.z;

            // float dis_sim = 1.0 - fmaxf(0.0f, 1.0f - norm(origin - x_world)/norm(camera_center - x_world));
            // cos_sim = cos_sim * dis_sim;

            float3 vec = camera_center - origin;


            float coor_a = dot(vec, basis_a);
            float coor_b = dot(vec, basis_b); 

            int region = ((signf(coor_b) + 1) + (signf(coor_a) + 1) / 2);


            atomicAdd(cur_score + region, cos_sim);
            // atomicAdd(score+(batch_idx*num_candidate+k) * 4 + region, cos_sim);

            cur_score = cur_score + 4;
        }

        // atomicAdd(score+(batch_idx*num_candidate+blockIdx.y) * 4 + region, cos_sim);

        taskIdx += total_thread;
    }
}


void pixel_level_neighbor_ranking_render(
    at::Tensor candidate_neighbors, 
    at::Tensor proj_mat,
    at::Tensor nei_cam_centers,
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, 
    at::Tensor up_axis,
    at::Tensor score, int num_thread,
    int height, int width)
{
    int batch_size = rays_o.size(0);
    int num_sample = z_vals.size(1);
    int num_camera = proj_mat.size(0);
    int num_candidate = candidate_neighbors.size(1);

    // PinholeCameraManager cameras((Intrinsic*)ks.contiguous().data_ptr<float>(), 
    //                             (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    // dim3 n_blocks = dim3(min(65535, (batch_size*num_sample + num_thread - 1) / num_thread), num_candidate);

    int n_blocks = min(65535, (batch_size*num_sample + num_thread - 1) / num_thread);

    pixel_level_neighbor_ranking_kernel<<<n_blocks, num_thread>>>(
        (int*)candidate_neighbors.contiguous().data_ptr<int>(),
        (float*)proj_mat.contiguous().data_ptr<float>(),
        (float3*)nei_cam_centers.contiguous().data_ptr<float>(),
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float*)z_vals.contiguous().data_ptr<float>(),
        (float3*)up_axis.contiguous().data_ptr<float>(),
        (float*)score.contiguous().data_ptr<float>(),
        height, width,
        num_candidate, num_sample, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


__device__ inline 
void sample_points_sparse_single_ray(
    float3 origin,
    float3 direction,
    int num_sample, // fine num
    int coarse_num,
    float* z_vals, 
    float* dists, 
    float3 block_corner, 
    float3 block_size,
    bool* occupied_gird,
    float near, float far, bool inv_z,
    bool background, 
    int3 log2dim)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    // printf("bound %f %f\n", bound.x, bound.y);


    if (bound.x == -1) return;

    // if (bound.x == 0.0) bound.x += 0.1f;

    // bound.x = fmaxf(bound.x, near);



    // bound.y = fminf(bound.y, far);

    // near = bound.x;
    // far = bound.y;

    float inv_near, inv_far;
    if (inv_z)
    {
        inv_near = 1.0f / fmaxf(near, bound.x);
        inv_far = 1.0f / fminf(far, bound.y);
    }else{
        near = 1.0f / fmaxf(near, bound.x);
        far = 1.0f / fminf(far, bound.y);
    }


    // printf("near %f far %f bound.x %f bound.y %f inv_near %f inv_far %f\n",
    // near, far, bound.x, bound.y, inv_near, inv_far);

    int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;


    float temp_len = 0.0f;
    bool flag = false;
    
    // while((!dda.terminate()) && (dda.t.x < far))
    while(!dda.terminate())
    {

        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        // uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = fmax(dda.t.y - dda.t.x, 0.0f);
            flag = true;
            temp_len += len;

        }else{
            if (flag == true) // 上一次是occupied 
            {
                flag = false;
                total_length += temp_len;
                count++;
                temp_len = 0.0f;
            }
        }

        dda.step();
    }

    if (flag == true) // 上一次是occupied 
    {
        flag = false;
        total_length += temp_len;
        count++;
        temp_len = 0.0f;
    }

    // printf("total_length %f count %d\n", total_length, count);


    if (count == 0) 
    {
        if (background == false)
        {
            return;
        }
        // no intersection with any grid    
        if (inv_z)
        {
            for (int i=0; i<num_sample+coarse_num; i++)
            {
                float s = 1.0f * i / (num_sample+coarse_num - 1);
                z_vals[i] = 1.0 / (inv_near * (1-s) + inv_far * s);
            }
        }else{
            for (int i=0; i<num_sample+coarse_num; i++)
            {
                float s = 1.0f * i / (num_sample+coarse_num - 1);
                z_vals[i] = near * (1 - s) + far * s; 
            } 
        }

        return;
    }else{
        if (inv_z)
        {
            for (int i=0; i<coarse_num; i++)
            {
                float s = 1.0f * i / (coarse_num - 1);
                z_vals[i] = 1.0 / (inv_near * (1-s) + inv_far * s);
            }
        }else{
            for (int i=0; i<coarse_num; i++)
            {
                float s = 1.0f * i / (coarse_num - 1);
                z_vals[i] = near * (1 - s) + far * s;
            }    
        }
        z_vals = z_vals + coarse_num;
    }

    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    int left_sample = num_sample;
    int sample_count = 0;

    float temp_near=0.0f;

    while(!dda.terminate())
    // while((!dda.terminate()) && (dda.t.x < far))
    {
        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;


        if (occupied_gird[n])
        {

            if (flag == false)
            {
                temp_near = dda.t.x;
            }
            float len = fmax(dda.t.y - dda.t.x, 0.0f);
            flag = true;
            temp_len += len;

        }else{
            if (flag == true && temp_len > 0)
            {
                flag = false;

                int num = min(max((int)(num_sample * temp_len / total_length), 1), left_sample);
                if (sample_count == count - 1) 
                {
                    num = left_sample;
                }

                uniform_sample_bound_v3(z_vals+num_sample-left_sample, 
                                        dists+num_sample-left_sample,
                                        temp_near, temp_near+temp_len, num);

                temp_len = 0.0f;
                left_sample = left_sample-num;
                sample_count++;                
            }
        }
        dda.step();
    }

    if (flag == true && temp_len > 0)
    {
        flag = false;

        int num = min(max((int)(num_sample * temp_len / total_length), 1), left_sample);
        if (sample_count == count - 1) 
        {
            num = left_sample;
        }

        uniform_sample_bound_v3(z_vals+num_sample-left_sample, 
                                dists+num_sample-left_sample,
                                temp_near, temp_near+temp_len, num);

        temp_len = 0.0f;
        left_sample = left_sample-num;
        sample_count++;                
    }
}


__global__ 
void sample_points_sparse_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample, 
    int coarse_num,
    float* z_vals,
    float* dists,
    float* block_corner,
    float* block_size,
    bool* occupied_gird,
    float near, float far, 
    int log2dim_x, int log2dim_y, int log2dim_z,
    bool inv_z, bool background,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        sample_points_sparse_single_ray(rays_o[taskIdx], rays_d[taskIdx], num_sample, coarse_num,
                                z_vals+taskIdx*(num_sample+coarse_num), dists+taskIdx*(num_sample+coarse_num),
                                make_float3(block_corner[0], block_corner[1], block_corner[2]), 
                                make_float3(block_size[0],block_size[1],block_size[2]), 
                                occupied_gird, near, far, inv_z, background,
                                make_int3(log2dim_x,log2dim_y,log2dim_z));
                                

        taskIdx += total_thread;
    }
}




void sample_points_grid_render(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor &dists,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird,
    float near, float far,
    int log2dim_x, int log2dim_y, int log2dim_z, bool inv_z, bool background)
{
    int batch_size = rays_o.size(0);
    int coarse_num = 64;
    int num_sample = z_vals.size(1) - coarse_num;


    sample_points_sparse_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        num_sample, coarse_num,
        z_vals.contiguous().data_ptr<float>(),
        dists.contiguous().data_ptr<float>(),
        (float*)block_corner.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(),
        (bool*)occupied_gird.contiguous().data_ptr<bool>(),
        near, far,
        log2dim_x, log2dim_y, log2dim_z, inv_z, background,
        batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}
