#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/ATen.h>
#include <cassert>
#include <cuda.h>
#include "cuda_fp16.h"
#include "macros.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include "cutil_math.h"
#include "cuda_utils.h"
#include "dda.h"
#include "decoder.h"
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <ATen/cuda/CUDABlas.h>


#define MAX_PTS_BLOCKS 4
#define INF_INTERSECTION 10000000.0f

__device__ inline
uint32_t hash(int3 loc, int hashmap_size)
{
    constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };
    uint32_t result = 0;
    result ^= loc.x * primes[0];
    result ^= loc.y * primes[1];
    result ^= loc.z * primes[2];

    return (hashmap_size - 1) & result;
}

__device__ inline  
void linear_weight(float* weight, float3 offset)
{
    weight[0] = (1-offset.x) * (1-offset.y) * (1-offset.z);
    weight[1] = (1-offset.x) * (1-offset.y) * offset.z;
    weight[2] = (1-offset.x) * offset.y * (1-offset.z);
    weight[3] = (1-offset.x) * offset.y * offset.z;
    weight[4] = offset.x * (1-offset.y) * (1-offset.z);
    weight[5] = offset.x * (1-offset.y) * offset.z;
    weight[6] = offset.x * offset.y * (1-offset.z);
    weight[7] = offset.x * offset.y * offset.z;
}

__device__ inline 
void hash_feature_index(uint32_t* indices, int3 bottom_left_idx, int hashmap_size)
{
    indices[0] = hash(bottom_left_idx + make_int3(0,0,0), hashmap_size);
    indices[1] = hash(bottom_left_idx + make_int3(0,0,1), hashmap_size);
    indices[2] = hash(bottom_left_idx + make_int3(0,1,0), hashmap_size);
    indices[3] = hash(bottom_left_idx + make_int3(0,1,1), hashmap_size);
    indices[4] = hash(bottom_left_idx + make_int3(1,0,0), hashmap_size);
    indices[5] = hash(bottom_left_idx + make_int3(1,0,1), hashmap_size);
    indices[6] = hash(bottom_left_idx + make_int3(1,1,0), hashmap_size);
    indices[7] = hash(bottom_left_idx + make_int3(1,1,1), hashmap_size);
}

__device__ inline 
void hash_features(float2* hashed_features, half2* level_feature, int3 bottom_left_idx, int hashmap_size)
{
    uint32_t indices[8];
    hash_feature_index(indices, bottom_left_idx, hashmap_size);
    #pragma unroll
    for (int i=0; i<8; i++)
    {
        hashed_features[i].x = __half2float(level_feature[indices[i]].x);
        hashed_features[i].y = __half2float(level_feature[indices[i]].y);
    }
}

template<int N_LEVELS>
__device__ inline 
void get_multilevel_features(
    float3 pts, //  should be inside [-1,1]
    int3* resolution, // num_levels
    half2* feature_tables, // num_levels x hashmap_size
    int hashmap_size,
    float* features)
{
    #pragma unroll
    for (int i=0; i<2*N_LEVELS; i++)
    {
        features[i] = 0.0f;
    }

    #pragma unroll
    for (int i=0; i<N_LEVELS; i++)
    {
        int3 level_resolution = resolution[i];
        float3 voxel_min_vertex = pts * make_float3(level_resolution-1);
        int3 bottom_left_idx = make_int3(voxel_min_vertex);
        float3 offset = voxel_min_vertex - make_float3(bottom_left_idx);

        float weight[8];
        linear_weight(weight, offset);
        float2 neighbor_features[8];
        hash_features(neighbor_features, feature_tables+i*hashmap_size, bottom_left_idx, hashmap_size);

        #pragma unroll
        for (int j=0; j<8; j++)
        {
            features[i*2+0] += weight[j] * neighbor_features[j].x;
            features[i*2+1] += weight[j] * neighbor_features[j].y;
        }
    }
}

__device__ inline 
void cal_accumulate_weight(float sigma, float t, float &weight, float &transparency)
{
    float alpha = 1.0f - expf(-1.0f * sigma * t);
    weight = transparency * alpha;
    transparency = transparency * (1.0f - alpha);
}

// ray block tracing 
__global__ 
void ray_block_intersection_kernel(
    float3* rays_o,
    float3* rays_d,
    float3* block_corners, 
    float3* block_sizes, 
    float2* intersections, // near, far shape batch_size x num_block
    int num_block, // total number of sparse blocks
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    int bidx = blockIdx.y;

    float3 half_size = block_sizes[bidx] / 2.0f;
    float3 block_center = block_corners[bidx] + half_size;
    while (taskIdx < batch_size)
    {

        float2 bound = RayAABBIntersection(rays_o[taskIdx], rays_d[taskIdx], block_center, half_size);
        if (bound.x == -1)
        {
            bound = make_float2(INF_INTERSECTION, INF_INTERSECTION);
        }
        intersections[taskIdx * num_block + bidx] = bound;

        taskIdx += total_thread;
    }
}

__host__ 
void ray_block_intersection(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor &intersections)
{
    int batch_size = rays_d.size(0);
    int num_block = block_corners.size(0); 
    ray_block_intersection_kernel<<<dim3(NUM_BLOCK(batch_size), num_block), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        (float2*)intersections.contiguous().data_ptr<float>(),
        num_block, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}


// sample points 
__device__ inline 
void count_seg(float3 origin, float3 direction,
               float2 t_start, 
               float3 block_corner, 
               float3 block_size, 
               bool* block_grid_occupied,
               int3 grid_log2dim,
               int &num_seg, float &total_length)
{
    int3 grid_resolution = make_int3(1 << grid_log2dim.x, 1 << grid_log2dim.y, 1 << grid_log2dim.z);

    DDASatateScene_v2 dda;
    float3 grid_size = block_size / make_float3(grid_resolution);
    dda.init(origin-block_corner, direction, t_start, grid_resolution, grid_size);

    while(!dda.terminate())
    {
        dda.next();

        int3 grid_loc = dda.current_tile;

        int grid_offset = (grid_loc.x << (grid_log2dim.y + grid_log2dim.z)) | (grid_loc.y << grid_log2dim.z) | grid_loc.z;
        if(block_grid_occupied[grid_offset])
        {
            // printf("grid_loc %d %d %d grid_log2dim %d %d %d\n", grid_loc.x, grid_loc.y, grid_loc.z,
            //         grid_log2dim.x, grid_log2dim.y, grid_log2dim.z);
            float len = dda.t.y - dda.t.x;
            if (len > 0)
            {
                total_length += len;
                num_seg +=1;
            }
        }
        dda.step();
    }
}


__device__ inline 
void block_sample_points(float3 origin, float3 direction,
                   float2 t_start, 
                   float3 block_corner, 
                   float3 block_size, 
                   bool* block_grid_occupied,
                   int3 grid_log2dim,
                   int num_seg, float total_length,
                   int num_sample, int &count, 
                   float* z_vals, float* dists,
                   int &num)
{
    int3 grid_resolution = make_int3(1 << grid_log2dim.x, 1 << grid_log2dim.y, 1 << grid_log2dim.z);

    DDASatateScene_v2 dda;
    float3 grid_size = block_size / make_float3(grid_resolution);
    dda.init(origin-block_corner, direction, t_start, grid_resolution, grid_size);
    while(!dda.terminate())
    {
        dda.next();

        int3 grid_loc = dda.current_tile;
        int grid_offset = (grid_loc.x << (grid_log2dim.y + grid_log2dim.z)) | (grid_loc.y << grid_log2dim.z) | grid_loc.z;
        if(block_grid_occupied[grid_offset])
        {
            float len = dda.t.y - dda.t.x;
            if (len > 0)
            {
                float near = dda.t.x;
                float far = dda.t.y;
                int n = min(max((int)(len / total_length * num_sample), 1), num_sample-num);
                if (count == num_seg-1) n = num_sample-num;
                if (n > 0)
                    uniform_sample_bound_v3(z_vals+num, dists+num, near, far, n);
                num = num + n;
                count++;
            }
        }
        dda.step();
    }
}

__global__ 
void samplepoints_kernel(
    float3* rays_o, float3* rays_d,
    float3* block_corners, // num_block x 3
    float3* block_sizes, // num_block x 3
    bool* grid_occupied,
    uint64_t* grid_starts,
    int3* grid_log2dim,
    int num_sample, int num_block,
    int* tracing_blocks, // B x num_block
    float2* intersections, // B x num_block
    int* tracing_idx, // B record 
    float* z_start, 
    float* z_vals, // B x num_sample x 1 
    float* dists,
    int batch_size) 
{

    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {

        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];
        int* cur_tracing_blocks = tracing_blocks + taskIdx * num_block;
        float2* cur_intersections = intersections + taskIdx * num_block;
        float* cur_z_vals = z_vals+taskIdx*num_sample;
        float* cur_dists = dists+taskIdx*num_sample;


        // points must inside the first block 
        int step =  tracing_idx[taskIdx];
        float2 bound = make_float2(-1.0f,-1.0f); 
        float2 t_start;
        t_start = make_float2(z_start[taskIdx],0);

        while (step < num_block)
        {
            int bidx = cur_tracing_blocks[step];
            bound = cur_intersections[bidx];
            // printf("bidx %d bound %f %f step %d t_start: %f %f\n",
            //         bidx, bound.x, bound.y, step, t_start.x, t_start.y);
            if (bound.x == INF_INTERSECTION)
            {
                // printf("bound %f\n", bound.x);
                break;
            }
            if (t_start.x >= bound.y) 
            {
                step++;
                continue;
            }
            if (step == 0) t_start.x = bound.x;

            int num_seg = 0;
            float total_length = 0.0f;
            count_seg(origin, direction, t_start, block_corners[bidx],
                    block_sizes[bidx], grid_occupied+grid_starts[bidx],
                    grid_log2dim[bidx], num_seg, total_length);
            // print("origin %f %f %f direction %f %f %f\n", 
            //         origin.x, origin.y, origin.z, 
            //         direction.x, direction.y, direction.z);
            // printf("step %f t_start %f num_seg %d total_length %f\n", step, t_start, num_seg, total_length);

            if (num_seg == 0)
            {
                t_start.x = bound.y;
                step++;
                continue;
            }
            int num = 0, count = 0;
            block_sample_points(origin, direction, t_start, block_corners[bidx],
                    block_sizes[bidx], grid_occupied+grid_starts[bidx],
                    grid_log2dim[bidx], num_seg, total_length, num_sample, count, 
                    cur_z_vals, cur_dists, num);   
            t_start.x = bound.y;
            step++;
            // printf("finished sampling~\n");
            break;       
        }
        tracing_idx[taskIdx] = step;
        z_start[taskIdx] = t_start.x;

        taskIdx += total_thread;
    }
}

__host__ 
void sample_points(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor tracing_blocks,
    at::Tensor intersections, 
    at::Tensor tracing_idx,
    at::Tensor &z_start, 
    at::Tensor &z_vals,
    at::Tensor &dists)
{
    int batch_size = rays_d.size(0);
    int num_block = block_corners.size(0);
    int num_sample = z_vals.size(1);

    samplepoints_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        (bool*)grid_occupied.contiguous().data_ptr<bool>(),
        (uint64_t*)grid_starts.contiguous().data_ptr<int64_t>(),
        (int3*)grid_log2dim.contiguous().data_ptr<int>(),
        num_sample, num_block, 
        (int*)tracing_blocks.contiguous().data_ptr<int>(),
        (float2*)intersections.contiguous().data_ptr<float>(),
        tracing_idx.contiguous().data_ptr<int>(),
        z_start.contiguous().data_ptr<float>(),
        (float*)z_vals.contiguous().data_ptr<float>(), 
        (float*)dists.contiguous().data_ptr<float>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}

// allocate point 
/*
points
int4 bidx  最多在 4个 block的overlap区域
float z_val 深度 
*/
__device__ inline
void get_pts_block(float z_val, float2* intersections, int num_block, short* blocks_idxs)
{
    if (z_val == -1.0f) return;
    int index = 0;
    for (int i=0; i<num_block; i++)
    {
        float2 bound = intersections[i];
        if (z_val >= bound.x && z_val <= bound.y)
        {
            blocks_idxs[index++] = i;
        }
    }
}

__global__ 
void prepare_points_kernel(
    float* z_vals, // B x num_sample 
    bool* runing_mask, // B 
    short* block_idxs, // B x num_sample x 4 
    float2* intersections, // B x num_block x 2 
    int num_sample, 
    int num_blocks,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size * num_sample)
    { 
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;
        if (runing_mask[batch_idx])
        {
            get_pts_block(z_vals[batch_idx * num_sample + sample_idx],  
                        intersections + batch_idx * num_blocks, num_blocks,
                        block_idxs + batch_idx * num_sample * MAX_PTS_BLOCKS + sample_idx * MAX_PTS_BLOCKS);
        }

        taskIdx += total_thread;
    }
}

__host__ 
void prepare_points(at::Tensor z_vals, 
                    at::Tensor runing_mask, at::Tensor intersections, at::Tensor &block_idxs)
{
    int batch_size = z_vals.size(0);
    int num_sample = z_vals.size(1);
    int num_block = intersections.size(1);
    prepare_points_kernel<<<NUM_BLOCK(batch_size * num_sample), NUM_THREAD>>>(
        z_vals.contiguous().data_ptr<float>(),
        runing_mask.contiguous().data_ptr<bool>(),
        (short*)block_idxs.contiguous().data_ptr<short>(),
        (float2*)intersections.contiguous().data_ptr<float>(),
        num_sample, num_block, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}

__host__
int sort_by_key(at::Tensor &keys_tensor, at::Tensor &values_tensor, at::Tensor &starts_tensor)
{
    int length = keys_tensor.size(0);
    short* keys = keys_tensor.data_ptr<short>();
    int* values = values_tensor.data_ptr<int>();
    int* starts = starts_tensor.data_ptr<int>();

    thrust::sort_by_key(thrust::device, keys, keys + length, values);

    thrust::pair<short*, int*> new_end = thrust::unique_by_key(thrust::device, keys, keys + length, starts);
    return new_end.second - starts;
}


template<int N_LEVELS>
__global__ 
void pts_inference_kernel(
    float3* rays_o, float3* rays_d,
    float* z_vals, // batch_size x num_sample
    float* dists,
    short* block_idxs,
    half2* features_tables, // num_block x ...
    float* params, // num_block x ..
    int3* resolution, // num_block x ...
    bool* grid_occupied, 
    uint64_t* grid_starts, // cus each grid may has its own shape 
    int3* grid_log2dim,  // num_blocks x 3 
    int hashmap_size,
    float3* block_corners, float3* block_sizes,
    float3* output_diffuse, float3* output_specular, float* output_alpha,
    int batch_size, int num_sample)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size * num_sample)
    { 
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;

        // if (sample_idx == num_sample - 1)
        // {
        //     // skip the last points 
        //     taskIdx += total_thread;
        //     continue;
        // }

        short* cur_block_idxs = block_idxs + taskIdx * MAX_PTS_BLOCKS;

        float3 diffuse = make_float3(0,0,0), specular=make_float3(0,0,0);
        float alpha = 0.0f, weight = 0.0f;
        #pragma unroll
        for (int i=0; i<MAX_PTS_BLOCKS; i++)
        {
            short bidx = cur_block_idxs[i];
            if (bidx == -1) break;

            int3 log2dim = grid_log2dim[bidx];
            int3 grid_resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
            bool* block_grid_occupied = grid_occupied + grid_starts[bidx];
            float3 block_corner = block_corners[bidx];
            float3 block_size = block_sizes[bidx];

            float3 origin = rays_o[batch_idx];
            float3 direction = rays_d[batch_idx];
            float z_val = z_vals[taskIdx]; float dist = dists[taskIdx];
            float3 pts = origin + z_val * direction;
            // [0,1]
            pts = (pts - block_corner) / block_size;

            // compute weight 
            float3 dis = (0.5f - fabs(pts - 0.5f)) * block_size;
            // float w = dis.x * dis.y * dis.z;
            float w;
            if (dis.x != 0 && dis.z != 0)
            {   
                w = dis.x * dis.z;
            }else if (dis.x != 0)
            {
                w = dis.x;
            }else if (dis.z != 0)
            {
                w = dis.z;
            }else{
                w = 0;
            }

            int3 grid_loc = make_int3(pts * make_float3(grid_resolution));
            grid_loc = clamp(grid_loc, make_int3(0,0,0), grid_resolution-1);
            int grid_offset = (grid_loc.x << (log2dim.y + log2dim.z)) | (grid_loc.y << log2dim.z) | grid_loc.z;

            if(block_grid_occupied[grid_offset])
            {
                float features[2*N_LEVELS];

                // [0,1] / 2 + 0.25 -> [0.25, 0.75] for foreground
                get_multilevel_features<N_LEVELS>(pts / 2.0f + 0.25f, 
                                        resolution + bidx * N_LEVELS, 
                                        features_tables + bidx * (N_LEVELS * hashmap_size),
                                        hashmap_size, features);
                Decoder decoder(params+bidx*PARAMSIZE);

                float3 pts_diffuse, pts_specular; float pts_sigma;
                decoder.inference(features, direction, pts_diffuse, pts_specular, pts_sigma);
                // blend
                float pts_alpha = 1.0f - expf(-1.0f * pts_sigma * dist * norm(direction));
                diffuse = diffuse + w * pts_alpha * pts_diffuse;
                specular = specular + w * pts_alpha * pts_specular;
                alpha = alpha + w * pts_alpha;

                weight += w;
            }else{
                weight += w;
            }            
        }

        if(weight > 0)
        {
            diffuse /= weight;
            specular /= weight;
            alpha /= weight;
        }

        output_diffuse[taskIdx] = diffuse;
        output_specular[taskIdx] = specular;
        output_alpha[taskIdx] = alpha;

        taskIdx += total_thread;
    }
}

__host__ 
void pts_inference(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor z_vals, at::Tensor dists,
    at::Tensor block_idxs,
    at::Tensor features_tables, at::Tensor params,
    at::Tensor resolution, 
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor &diffuse, at::Tensor &specular, at::Tensor &alpha)
{
    int batch_size = rays_d.size(0);
    int num_sample = z_vals.size(1);
    int hashmap_size = features_tables.size(2);

    pts_inference_kernel<16><<<NUM_BLOCK_256(batch_size*num_sample), NUM_THREAD_256>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        z_vals.contiguous().data_ptr<float>(),
        dists.contiguous().data_ptr<float>(),
        (short*)block_idxs.contiguous().data_ptr<short>(),
        (half2*)features_tables.contiguous().data_ptr<at::Half>(),
        (float*)params.contiguous().data_ptr<float>(),
        (int3*)resolution.contiguous().data_ptr<int>(),
        (bool*)grid_occupied.contiguous().data_ptr<bool>(),
        (uint64_t*)grid_starts.contiguous().data_ptr<int64_t>(),
        (int3*)grid_log2dim.contiguous().data_ptr<int>(),
        hashmap_size,
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        (float3*)diffuse.contiguous().data_ptr<float>(),
        (float3*)specular.contiguous().data_ptr<float>(),
        alpha.contiguous().data_ptr<float>(), batch_size, num_sample);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}

__global__ 
void accumulate_kernel(
    float3* pts_diffuse, // batch_size x num_sample 
    float3* pts_specular,
    float* pts_alpha, 
    float* transparency,
    float* z_vals,
    int num_sample,
    int batch_size,
    float3* output_diffuse,
    float3* output_specular,
    float* output_depth)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {   
        float T = transparency[taskIdx];
        if (T < 0.00001f) 
        {
            taskIdx += total_thread;
            continue;
        }

        float3* cur_pts_diffuse = pts_diffuse + taskIdx * num_sample;
        float3* cur_pts_specular = pts_specular + taskIdx * num_sample;
        float* cur_pts_alpha = pts_alpha + taskIdx * num_sample;
        float* cur_z_vals = z_vals + taskIdx * num_sample;

        float3 diffuse = output_diffuse[taskIdx];
        float3 specular = output_specular[taskIdx];
        float depth = output_depth[taskIdx];

        for (int i=0; i<num_sample; i++)
        {
            // printf("i %d diffuse %f %f %f T %f alpha %f\n",
            //         i, diffuse.x, diffuse.y, diffuse.z,
            //         T, cur_pts_alpha[i]);
            diffuse += T * cur_pts_diffuse[i];
            specular += T * cur_pts_specular[i];
            depth += T * cur_pts_alpha[i] * cur_z_vals[i];
            T = T * (1 - cur_pts_alpha[i]);
        }
        transparency[taskIdx] = T;
        output_diffuse[taskIdx] = diffuse;
        output_specular[taskIdx] = specular;
        output_depth[taskIdx] = depth;

        taskIdx += total_thread;
    }
}

__host__ 
void accumulate_color(
    at::Tensor pts_diffuse,
    at::Tensor pts_specular,
    at::Tensor pts_alpha,
    at::Tensor &transparency,
    at::Tensor z_vals,
    at::Tensor &diffuse,
    at::Tensor &specular,
    at::Tensor &depth)
{
    int batch_size = z_vals.size(0);
    int num_sample = z_vals.size(1);

    accumulate_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)pts_diffuse.contiguous().data_ptr<float>(),
        (float3*)pts_specular.contiguous().data_ptr<float>(),
        (float*)pts_alpha.contiguous().data_ptr<float>(),
        (float*)transparency.contiguous().data_ptr<float>(),
        (float*)z_vals.contiguous().data_ptr<float>(),
        num_sample, batch_size,
        (float3*)diffuse.contiguous().data_ptr<float>(),
        (float3*)specular.contiguous().data_ptr<float>(),
        (float*)depth.contiguous().data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}

__device__ inline 
bool tracing(float3 origin, float3 direction,
             float2 t_start, 
             float3 block_corner, 
             float3 block_size, 
             bool* block_grid_occupied,
             int3 grid_log2dim)
{
    int3 grid_resolution = make_int3(1 << grid_log2dim.x, 1 << grid_log2dim.y, 1 << grid_log2dim.z);

    DDASatateScene_v2 dda;
    float3 grid_size = block_size / make_float3(grid_resolution);
    dda.init(origin-block_corner, direction, t_start, grid_resolution, grid_size);

    while(!dda.terminate())
    {
        dda.next();

        int3 grid_loc = dda.current_tile;
        int grid_offset = (grid_loc.x << (grid_log2dim.y + grid_log2dim.z)) | (grid_loc.y << grid_log2dim.z) | grid_loc.z;
        if(block_grid_occupied[grid_offset])
        {
           return true;
        }
        dda.step();
    }
    return false;
}

// improved sampling 
__global__ 
void ray_firsthit_block_kernel(
    float3* rays_o, float3* rays_d,
    float3* block_corners, // num_block x 3
    float3* block_sizes, // num_block x 3
    bool* grid_occupied,
    uint64_t* grid_starts,
    int3* grid_log2dim,
    int* tracing_blocks, // B x num_block
    float2* intersections, // B x num_block
    short* hit_blockIdxs, // B 
    int num_block,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];
        int* cur_tracing_blocks = tracing_blocks + taskIdx * num_block;
        float2* cur_intersections = intersections + taskIdx * num_block;

        float dis = 10000000.0f;
        int last_bidx = -1;
        for (int i=0; i<num_block; i++)
        {
            int bidx = cur_tracing_blocks[i];
            float2 bound = cur_intersections[bidx];
            if (bound.x == INF_INTERSECTION) break;

            bool ret = tracing(origin, direction, bound, block_corners[bidx], block_sizes[bidx], 
                                grid_occupied+grid_starts[bidx], grid_log2dim[bidx]);
            
            if (ret && dis > bound.y)
            {
                hit_blockIdxs[taskIdx] = bidx;
                dis = bound.y;
            }
            last_bidx = bidx;
        }

        if (last_bidx != -1 && hit_blockIdxs[taskIdx] == -1)
        {
            hit_blockIdxs[taskIdx] = last_bidx;
        }
        taskIdx += total_thread;
    }
}

__host__ 
void ray_firsthit_block(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor tracing_blocks,
    at::Tensor intersections, 
    at::Tensor &hit_blockIdxs)
{
    int batch_size = rays_d.size(0);
    int num_block = block_corners.size(0);

    ray_firsthit_block_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        (bool*)grid_occupied.contiguous().data_ptr<bool>(),
        (uint64_t*)grid_starts.contiguous().data_ptr<int64_t>(),
        (int3*)grid_log2dim.contiguous().data_ptr<int>(),
        (int*)tracing_blocks.contiguous().data_ptr<int>(),
        (float2*)intersections.contiguous().data_ptr<float>(),
        (short*)hit_blockIdxs.contiguous().data_ptr<short>(),
        num_block, batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__global__ 
void inverse_z_sampling_kernel(
    float2* intersections, // B x num_block
    short* related_bidx, // B x 
    int num_sample, 
    int num_block,
    float sample_range, 
    float* z_vals,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        float2* cur_intersections = intersections + taskIdx * num_block;
        float* cur_z_vals = z_vals + taskIdx * num_sample;

        short bidx = related_bidx[taskIdx];

        if (bidx == -1)
        {
            taskIdx += total_thread;   
            continue;
        }

        float2 bound = cur_intersections[bidx];

        if (bound.x != INF_INTERSECTION)
        {
            float near = bound.y;
            float far = near + sample_range;
            inverse_z_sample_bound(cur_z_vals, near, far, num_sample);
        }

        taskIdx += total_thread;   
    }
}

__host__ 
void inverse_z_sampling(
    at::Tensor intersections,  at::Tensor related_bidx,
    at::Tensor &z_vals,  float sample_range)
{
    int batch_size = intersections.size(0);
    int num_block = intersections.size(1);
    int num_sample = z_vals.size(1); 
    inverse_z_sampling_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float2*)intersections.contiguous().data_ptr<float>(),
        related_bidx.contiguous().data_ptr<short>(),
        num_sample, num_block, sample_range,
        z_vals.contiguous().data_ptr<float>(), batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


template<int N_LEVELS>
__global__ 
void bg_pts_inference_kernel(
    float3* rays_o, float3* rays_d,
    float* z_vals, // B x num_sample
    half2* features_tables, // num_block x ...
    float* params, // num_block x ..
    float3* block_corners, float3* block_sizes,
    int3* resolution,
    short* outgoing_bidxs, // B x 4
    float* blend_weights,
    float3* output_diffuse, // B x num_sample
    float3* output_specular,
    float* output_alpha,
    int hashmap_size,
    int batch_size, int num_sample)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size * num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;


        float3 origin = rays_o[batch_idx];
        float3 direction = rays_d[batch_idx];
        float z_val = z_vals[taskIdx];

        float sample_step;
        if (sample_idx == num_sample - 1) sample_step = 10000000.0f;
        else sample_step = z_vals[taskIdx+1] - z_val;

        short* cur_bidxs = outgoing_bidxs+batch_idx*MAX_PTS_BLOCKS;
        float* cur_blend_weights = blend_weights+batch_idx*MAX_PTS_BLOCKS;


        float3 diffuse = make_float3(0,0,0), specular=make_float3(0,0,0);
        float alpha = 0.0f, weight = 0.0f;
        #pragma unroll 
        for (int i=0; i<MAX_PTS_BLOCKS; i++)
        {
            short bidx = cur_bidxs[i];
            if (bidx == -1) break;
            float3 block_corner = block_corners[bidx];
            float3 block_size = block_sizes[bidx];

            float pts[3];
            pts[0] = origin.x + z_val * direction.x;
            pts[1] = origin.y + z_val * direction.y;
            pts[2] = origin.z + z_val * direction.z;
            pts[0] = 2.0f * (pts[0] - block_corner.x) / block_size.x - 1.0f;
            pts[1] = 2.0f * (pts[1] - block_corner.y) / block_size.y - 1.0f;
            pts[2] = 2.0f * (pts[2] - block_corner.z) / block_size.z - 1.0f;

            // compute weight 
            float w = cur_blend_weights[i];

            float abs_pts[3];
            float sign[3];
            #pragma unroll
            for (int j=0; j<3; j++)
            {
                abs_pts[j] = pts[j];
                sign[j] = 1.0f;
                if(pts[j] < 0) 
                {
                    abs_pts[j] = -1.0f * pts[j];
                    sign[j] = -1.0f;
                }
            }

            int infinity_loc = 0;
            float infinity_norm = abs_pts[0];
            #pragma unroll
            for (int j=1; j<3; j++)
            {
                if(abs_pts[j] > infinity_norm)
                {
                    infinity_norm = abs_pts[j];
                    infinity_loc = j;
                }
            }

            float temp = (2.0f - 1.0f / infinity_norm);
            float ratio = temp / infinity_norm;
            #pragma unroll
            for (int j=0; j<3; j++)
            {
                pts[j] *= ratio;
            }   

            // #pragma unroll
            // for (int j=0; j<3; j++)
            // {
            //     if (j == infinity_loc)
            //     {
            //         pts[j] = (2.0f - 1.0f / abs_pts[j]) * sign[j];
            //     }else{
            //         pts[j] = pts[j] / infinity_norm;
            //     }
            // }

            float3 contract_x = make_float3(pts[0],pts[1],pts[2]);


            float features[2*N_LEVELS];

            // [-2,2] +2 / 4  -> [0, 1] for backgroud
            get_multilevel_features<N_LEVELS>((contract_x + 2.0f) / 4.0f, 
                                    resolution + bidx * N_LEVELS, 
                                    features_tables + bidx * (N_LEVELS * hashmap_size),
                                    hashmap_size, features);
            Decoder decoder(params+bidx*PARAMSIZE);
            float3 pts_diffuse, pts_specular; float pts_sigma;
            decoder.inference(features, direction, pts_diffuse, pts_specular, pts_sigma);
            float pts_alpha = 1.0f - expf(-1.0f * pts_sigma * sample_step);

            diffuse = diffuse + w * pts_alpha * pts_diffuse;
            specular = specular + w * pts_alpha * pts_specular;
            alpha = alpha + w * pts_alpha;
            weight += w;
        }

        if(weight > 0)
        {
            diffuse /= weight;
            specular /= weight;
            alpha /= weight;
        }

        output_diffuse[taskIdx] = diffuse;
        output_specular[taskIdx] = specular;
        output_alpha[taskIdx] = alpha;

        taskIdx += total_thread;
    }
}


template<int N_LEVELS>
__global__ 
void bg_pts_inference_v2_kernel(
    float3* rays_o, float3* rays_d,
    float* z_vals, // B x num_sample
    half2* features_tables, // num_block x ...
    float* params, // num_block x ..
    float3* block_corners, float3* block_sizes,
    int3* resolution,
    short* bg_idxs, // B x 4
    int step,
    float3* output_diffuse, // B x num_sample
    float3* output_specular,
    float* output_alpha,
    int hashmap_size,
    int batch_size, int num_sample)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size * num_sample)
    {
        int batch_idx = taskIdx / num_sample;
        int sample_idx = taskIdx % num_sample;

        short bidx = bg_idxs[batch_idx*MAX_PTS_BLOCKS+step];
        if (bidx == -1) {
            taskIdx += total_thread;
            continue;
        }

        float3 origin = rays_o[batch_idx];
        float3 direction = rays_d[batch_idx];
        float z_val = z_vals[taskIdx];

        float sample_step;
        if (sample_idx == num_sample - 1) sample_step = 10000000.0f;
        else sample_step = z_vals[taskIdx+1] - z_val;

        float3 diffuse = make_float3(0,0,0), specular=make_float3(0,0,0);
        float alpha = 0.0f, weight = 0.0f;


        float3 block_corner = block_corners[bidx];
        float3 block_size = block_sizes[bidx];

        float pts[3];
        pts[0] = origin.x + z_val * direction.x;
        pts[1] = origin.y + z_val * direction.y;
        pts[2] = origin.z + z_val * direction.z;
        pts[0] = 2.0f * (pts[0] - block_corner.x) / block_size.x - 1.0f;
        pts[1] = 2.0f * (pts[1] - block_corner.y) / block_size.y - 1.0f;
        pts[2] = 2.0f * (pts[2] - block_corner.z) / block_size.z - 1.0f;

        float abs_pts[3];
        float sign[3];
        #pragma unroll
        for (int j=0; j<3; j++)
        {
            abs_pts[j] = pts[j];
            sign[j] = 1.0f;
            if(pts[j] < 0) 
            {
                abs_pts[j] = -1.0f * pts[j];
                sign[j] = -1.0f;
            }
        }

        int infinity_loc = 0;
        float infinity_norm = abs_pts[0];
        #pragma unroll
        for (int j=1; j<3; j++)
        {
            if(abs_pts[j] > infinity_norm)
            {
                infinity_norm = abs_pts[j];
                infinity_loc = j;
            }
        }

        float temp = (2.0f - 1.0f / infinity_norm);
        float ratio = temp / infinity_norm;
        #pragma unroll
        for (int j=0; j<3; j++)
        {
            pts[j] *= ratio;
        }   


        // #pragma unroll
        // for (int j=0; j<3; j++)
        // {
        //     if (j == infinity_loc)
        //     {
        //         pts[j] = (2.0f - 1.0f / abs_pts[j]) * sign[j];
        //     }else{
        //         pts[j] = pts[j] / infinity_norm;
        //     }
        // }

        float3 contract_x = make_float3(pts[0],pts[1],pts[2]);

        float features[2*N_LEVELS];

        // [-2,2] +2 / 4  -> [0, 1] for backgroud
        get_multilevel_features<N_LEVELS>((contract_x + 2.0f) / 4.0f, 
                                resolution + bidx * N_LEVELS, 
                                features_tables + bidx * (N_LEVELS * hashmap_size),
                                hashmap_size, features);

        // for (int j=0; j<N_LEVELS; j++)
        // {
        //     features[j*2] *= weight_feature[j];
        //     features[j*2+1] *= weight_feature[j];
        // }

        Decoder decoder(params+bidx*PARAMSIZE);
        float3 pts_diffuse, pts_specular; float pts_sigma;
        decoder.inference(features, direction, pts_diffuse, pts_specular, pts_sigma);
        float pts_alpha = 1.0f - expf(-1.0f * pts_sigma * sample_step);

        output_diffuse[taskIdx] = pts_alpha * pts_diffuse;
        output_specular[taskIdx] = pts_alpha * pts_specular;
        output_alpha[taskIdx] = pts_alpha;

        taskIdx += total_thread;
    }
}

__host__ 
void bg_pts_inference_v2(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, at::Tensor bg_idxs,
    int step,
    at::Tensor block_corners, at::Tensor block_sizes, 
    at::Tensor resolution,
    at::Tensor features_tables, at::Tensor params,
    at::Tensor &diffuse, at::Tensor &specular, at::Tensor &alpha)
{
    int batch_size = rays_d.size(0);
    int num_sample = z_vals.size(1);
    int hashmap_size = features_tables.size(2);

    bg_pts_inference_v2_kernel<16><<<NUM_BLOCK_256(batch_size*num_sample), NUM_THREAD_256>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        z_vals.contiguous().data_ptr<float>(),
        (half2*)features_tables.contiguous().data_ptr<at::Half>(),
        (float*)params.contiguous().data_ptr<float>(),
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        (int3*)resolution.contiguous().data_ptr<int>(),
        bg_idxs.contiguous().data_ptr<short>(), step,
        (float3*)diffuse.contiguous().data_ptr<float>(),
        (float3*)specular.contiguous().data_ptr<float>(),
        alpha.contiguous().data_ptr<float>(), 
        hashmap_size,batch_size,num_sample);

    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}



__host__ 
void bg_pts_inference(
    at::Tensor rays_o, at::Tensor rays_d, 
    at::Tensor z_vals, at::Tensor outgoing_bidxs,
    at::Tensor blend_weights,
    at::Tensor block_corners, at::Tensor block_sizes, 
    at::Tensor resolution,
    at::Tensor features_tables, at::Tensor params,
    at::Tensor &diffuse, at::Tensor &specular, at::Tensor &alpha)
{
    int batch_size = rays_d.size(0);
    int num_sample = z_vals.size(1);
    int hashmap_size = features_tables.size(2);

    bg_pts_inference_kernel<16><<<NUM_BLOCK_256(batch_size*num_sample), NUM_THREAD_256>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        z_vals.contiguous().data_ptr<float>(),
        (half2*)features_tables.contiguous().data_ptr<at::Half>(),
        (float*)params.contiguous().data_ptr<float>(),
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        (int3*)resolution.contiguous().data_ptr<int>(),
        outgoing_bidxs.contiguous().data_ptr<short>(),
        blend_weights.contiguous().data_ptr<float>(),
        (float3*)diffuse.contiguous().data_ptr<float>(),
        (float3*)specular.contiguous().data_ptr<float>(),
        alpha.contiguous().data_ptr<float>(), 
        hashmap_size,batch_size,num_sample);

    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__global__ 
void get_last_block_kernel(
    int* tracing_blocks,
    int* bidxs, 
    float2* intersections,
    int num_block,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {

        int* cur_tracing_blocks = tracing_blocks + taskIdx * num_block;
        float2* cur_intersections = intersections + taskIdx * num_block;

        int idx = -1;
        for (int i=0; i<num_block; i++)
        {
            int bidx = cur_tracing_blocks[i];
            float2 bound = cur_intersections[bidx];
            if (bound.x == INF_INTERSECTION)
            {
                break;
            }
            idx = bidx;
        }
        bidxs[taskIdx] = idx;
        taskIdx += total_thread;
    }
}

__host__ 
void get_last_block(
    at::Tensor tracing_blocks,
    at::Tensor &bidxs,
    at::Tensor intersections)
{
    int batch_size = intersections.size(0);
    int num_block = intersections.size(1);

    get_last_block_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        tracing_blocks.contiguous().data_ptr<int>(),
        bidxs.contiguous().data_ptr<int>(),
        (float2*)intersections.contiguous().data_ptr<float>(),
        num_block, batch_size);
    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}

__global__ 
void update_outgoing_bidx_kernel(
    float3* rays_o, float3* rays_d,
    float3* block_corners, float3* block_sizes, 
    int* tracing_blocks, // B x num_block 
    float2* intersections, // B x num_block 
    short* outgoing_bidxs, // B x 4
    float*  blend_weight, // B x 4 
    float ratio,
    bool skip, 
    int num_block, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {
        int* cur_tracing_blocks = tracing_blocks + taskIdx * num_block;
        float2* cur_intersections = intersections + taskIdx * num_block;

        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];

        float far = -1.0f;
        int index = 0;
        short outgonging_blocks[MAX_PTS_BLOCKS] = {-1,-1,-1,-1};
        for (int i=0; i<num_block; i++)
        {
            int bidx = cur_tracing_blocks[i];
            float2 bound = cur_intersections[bidx];
            if(bound.x == INF_INTERSECTION) break;

            if (!skip && (bound.x > far && far != -1.0f)) break;

            if (bound.y > far)
            {
                far = bound.y; 
                #pragma unroll 
                for (int j=0; j<MAX_PTS_BLOCKS; j++)
                {
                    outgonging_blocks[j] = -1;
                }
                index = 0;
                outgonging_blocks[index++] = bidx;
            }else if (bound.y == far)
            {
                outgonging_blocks[index++] = bidx;
            }
        }

        if(far == -1.0f) 
        {
            taskIdx += total_thread; 
            continue;
        }
        
        float3 pts = origin + far * direction;

        if (index == 1)
        {
            blend_weight[taskIdx*MAX_PTS_BLOCKS] = 1.0f;
            outgoing_bidxs[taskIdx*MAX_PTS_BLOCKS] = outgonging_blocks[0];
            taskIdx += total_thread;
            continue;
        }

        #pragma unroll 
        for(int i=0; i<MAX_PTS_BLOCKS; i++)
        {
            int bidx = outgonging_blocks[i];
            if (bidx == -1) break;

            // float sin_theta = (pts.y - origin.y) / norm(pts - origin);
            float3 block_corner = block_corners[bidx];
            float3 block_size = block_sizes[bidx];
            float3 p = clamp((pts - block_corner) / block_size, 0, 1);
            float3 dis = (0.5f - fabs(p - 0.5f)) * block_size;

            // if (dis.y < block_size.y * ratio)
            // {
            //     for (int j=0; j<MAX_PTS_BLOCKS; j++)
            //     {
            //         blend_weight[taskIdx*MAX_PTS_BLOCKS + j] = stand_blocks[j];
            //         outgoing_bidxs[taskIdx*MAX_PTS_BLOCKS + j] = stand_blendweight[j];
            //     }
            //     break;
            // }
            // 有一个面一定是0 

            // do not consider y axis 
            if (dis.x != 0 && dis.z != 0)
            {
                blend_weight[taskIdx*MAX_PTS_BLOCKS + i] = dis.x * dis.z;
                outgoing_bidxs[taskIdx*MAX_PTS_BLOCKS + i] = bidx;
            }else if (dis.x != 0)
            {
                blend_weight[taskIdx*MAX_PTS_BLOCKS + i] = dis.x;
                outgoing_bidxs[taskIdx*MAX_PTS_BLOCKS + i] = bidx;
            }else if (dis.z != 0)
            {
                blend_weight[taskIdx*MAX_PTS_BLOCKS + i] = dis.z;
                outgoing_bidxs[taskIdx*MAX_PTS_BLOCKS + i] = bidx;
            }else{
                blend_weight[taskIdx*MAX_PTS_BLOCKS + i] = 0.0f;
                outgoing_bidxs[taskIdx*MAX_PTS_BLOCKS + i] = bidx;
            }
            
        }

        taskIdx += total_thread;
    }
}

__host__ 
void update_outgoing_bidx(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor tracing_blocks,
    at::Tensor intersections,
    at::Tensor &outgoing_bidxs,
    at::Tensor &blend_weights, float ratio, bool skip)
{
    int batch_size = tracing_blocks.size(0);
    int num_block = tracing_blocks.size(1);

    update_outgoing_bidx_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        tracing_blocks.contiguous().data_ptr<int>(),
        (float2*)intersections.contiguous().data_ptr<float>(),
        outgoing_bidxs.contiguous().data_ptr<short>(),
        blend_weights.contiguous().data_ptr<float>(), ratio, skip,
        num_block, batch_size);

    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


// 按照当前视点选择 background
__global__ 
void update_outgoing_bidx_v2_kernel(
    float3* rays_o, float3* rays_d,
    float3* block_corners, float3* block_sizes, 
    int* tracing_blocks, // B x num_block 
    float2* intersections, // B x num_block 
    short* inside_bidxs, // B x 4
    float*  blend_weight, // B x 4 
    int num_block, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while(taskIdx < batch_size)
    {

        float3 origin = rays_o[taskIdx];

        short* cur_inside_bidxs = inside_bidxs + taskIdx * MAX_PTS_BLOCKS;
        float* cur_blend_weight = blend_weight + taskIdx * MAX_PTS_BLOCKS;
        int index = 0;
        
        for (int i=0; i<num_block; i++)
        {
            float3 block_corner = block_corners[i];
            float3 block_size = block_sizes[i];
            float3 loc = (origin - block_corner) / block_size;

            if (loc.x >= 0 && loc.x <= 1 &&
                loc.y >= 0 && loc.y <= 1 &&
                loc.z >= 0 && loc.z <= 1)
            {
                cur_inside_bidxs[index] = i;
                float3 dis = (0.5f - fabs(loc - 0.5f)) * block_size;
                cur_blend_weight[index] = dis.x * dis.y * dis.z;
                index++;
            }

        }

        taskIdx += total_thread;
    }
}


__host__ 
void update_outgoing_bidx_v2(
    at::Tensor rays_o, at::Tensor rays_d,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor tracing_blocks,
    at::Tensor intersections,
    at::Tensor &inside_bidxs,
    at::Tensor &blend_weights)
{
    int batch_size = tracing_blocks.size(0);
    int num_block = tracing_blocks.size(1);

    update_outgoing_bidx_v2_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        tracing_blocks.contiguous().data_ptr<int>(),
        (float2*)intersections.contiguous().data_ptr<float>(),
        inside_bidxs.contiguous().data_ptr<short>(),
        blend_weights.contiguous().data_ptr<float>(),
        num_block, batch_size);

    
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}

__global__ 
void process_occupied_grid_kernel(
    int bidx, int num_block, 
    float3* block_corners, float3* block_sizes,
    bool* grid_occupied,
    uint64_t* grid_starts,
    int3* grid_log2dim, 
    bool* tgt_grid_occupied,
    int total_grid)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    float3 block_corner = block_corners[bidx];
    float3 block_size = block_sizes[bidx];
    bool* block_grid_occupied = grid_occupied + grid_starts[bidx];
    int3 log2dim = grid_log2dim[bidx];
    int3 resolution = make_int3(1<<log2dim.x, 1<<log2dim.y, 1<<log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);
    while(taskIdx < total_grid)
    {
        if (block_grid_occupied[taskIdx] == false)
        {
            taskIdx += total_thread;
            continue;
        }
        int x = taskIdx / (resolution.y * resolution.z);
        int y = (taskIdx - x * (resolution.y * resolution.z)) / resolution.z;
        int z = (taskIdx - x * (resolution.y * resolution.z)) % resolution.z;

        int3 loc = make_int3(x, y, z);

        // float3 pts = (make_float3(loc) + 0.5f) * grid_size + block_corner;
        float3 pts = make_float3(loc) * grid_size + block_corner;

        float3 vertex[8] = {make_float3(0,0,0), make_float3(0,0,1), make_float3(0,1,0), make_float3(1,0,0), 
                            make_float3(0,1,1), make_float3(1,0,1), make_float3(1,1,0), make_float3(1,1,1)};

        for (int i=0; i<num_block; i++)
        {
            if (i == bidx) continue;
            float3 corner = block_corners[i];
            float3 size = block_sizes[i];

            #pragma unroll 
            for (int j=0; j<8; j++)
            {
                float3 p = (pts + vertex[j] * grid_size - corner) / size;
                if (p.x >= 0 && p.x < 1 && 
                    p.y >= 0 && p.y < 1 &&
                    p.z >= 0 && p.z < 1)
                {
                    bool* grid = tgt_grid_occupied + grid_starts[i];
                    int3 l2d = grid_log2dim[i];
                    int3 r = make_int3(1 << l2d.x, 1 << l2d.y, 1 << l2d.z);
                    int3 ijk = make_int3(p * make_float3(r));
                    int n = (ijk.x << (l2d.y + l2d.z)) | (ijk.y << l2d.z) | ijk.z;
                    grid[n] = true;
                }
            }
        }
        taskIdx += total_thread;
    }
}

__host__ 
void process_occupied_grid(
    int bidx, int total_grid,
    at::Tensor block_corners, at::Tensor block_sizes,
    at::Tensor grid_occupied, 
    at::Tensor grid_starts, 
    at::Tensor grid_log2dim,
    at::Tensor & tgt_grid_occupied)
{
    int num_block = block_corners.size(0);


    process_occupied_grid_kernel<<<NUM_BLOCK(total_grid), NUM_THREAD>>>(
        bidx, num_block, 
        (float3*)block_corners.contiguous().data_ptr<float>(),
        (float3*)block_sizes.contiguous().data_ptr<float>(),
        (bool*)grid_occupied.contiguous().data_ptr<bool>(),
        (uint64_t*)grid_starts.contiguous().data_ptr<int64_t>(),
        (int3*)grid_log2dim.contiguous().data_ptr<int>(),
        (bool*)tgt_grid_occupied.contiguous().data_ptr<bool>(), total_grid);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}