#ifndef SAMPLER__
#define SAMPLER__

#include "plyIO.h"
#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>
#include "block_data.h"

#define INF 100000000.0f

void sample_points_cuda(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    int num_sample, 
    at::Tensor &z_vals,
    DataBuffer buffer);

class Sampler
{
public:
    DataBuffer buffer;

    float3 bbox_min;
    float3 bbox_size;
    uint3 log2dim;

    Sampler(){}
    ~Sampler(){}

    void rebuild(at::Tensor vis, at::Tensor _log2dim)
    {
        buffer.free();

        log2dim = make_uint3(_log2dim.contiguous().data_ptr<int>()[0],
                             _log2dim.contiguous().data_ptr<int>()[1],
                             _log2dim.contiguous().data_ptr<int>()[2]);

        uint32_t total_size = (1u << log2dim.x) * (1u << log2dim.y) * (1u << log2dim.z);
        uint32_t size_mask = total_size >> 6;
        uint64_t* bitmask = new uint64_t[size_mask];
        for (int i=0; i<size_mask; i++) bitmask[i] = 0;

        bool* vis_ptr = vis.contiguous().data_ptr<bool>();
        for (int i=0; i<total_size; i++)
        {
            if (vis_ptr[i]) bitmask[i >> 6] |= uint64_t(1) << (i & 63);
        }

        buffer.create(bitmask, bbox_min, bbox_size, log2dim, size_mask);
        delete [] bitmask;
    }

    void build(at::Tensor _log2dim, at::Tensor block_corner, at::Tensor block_size, std::string model_path, at::Tensor &vis,
               bool init_out)
    {
        log2dim = make_uint3(_log2dim.contiguous().data_ptr<int>()[0],
                            _log2dim.contiguous().data_ptr<int>()[1],
                            _log2dim.contiguous().data_ptr<int>()[2]);

        bbox_size = make_float3(block_size.contiguous().data_ptr<float>()[0],
                                block_size.contiguous().data_ptr<float>()[1],
                                block_size.contiguous().data_ptr<float>()[2]);

        int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);

        uint32_t size_mask = (resolution.x * resolution.y * resolution.z) >> 6;

        uint64_t* bitmask = new uint64_t[size_mask];
        for (int i=0; i<size_mask; i++) bitmask[i] = 0;

        float3 grid_size = bbox_size / make_float3((float)resolution.x, (float)resolution.y, (float)resolution.z); 


        bbox_min = make_float3(block_corner.contiguous().data_ptr<float>()[0],
                               block_corner.contiguous().data_ptr<float>()[1],
                               block_corner.contiguous().data_ptr<float>()[2]);

        // printf("grid_size %f %f %f resolution: %d %d %d bbox_min %f %f %f\n", grid_size.x, grid_size.y, grid_size.z,
        //     resolution.x, resolution.y, resolution.z, bbox_min.x, bbox_min.y, bbox_min.z);

        float3 bbox_max = bbox_min + bbox_size;

        bool* vis_ptr = vis.contiguous().data_ptr<bool>();
        if (model_path != "")
        {
            std::vector<float3> vertices;
            std::vector<int3> faces;
            std::vector<float2> uv_array;
            read_plyFile(model_path, vertices, faces, uv_array);



            float3 geo_min = make_float3(INF, INF, INF);
            float3 geo_max = -1.0f * geo_min;
            for (int i=0; i<faces.size(); i++)
            {
                int3 vidx = faces[i];
                float3 A = vertices[vidx.x];
                float3 B = vertices[vidx.y];
                float3 C = vertices[vidx.z];

                float3 min_c = fminf(fminf(A, B), C);
                float3 max_c = fmaxf(fmaxf(A, B), C);

                float3 tri_center = (min_c + max_c) / 2.0f;
                float3 tri_size = (max_c - min_c) * 1.5f;  
                float3 half_trisize = tri_size / 2.0f;
                min_c = tri_center - half_trisize;
                max_c = tri_center + half_trisize;
        
                if (max_c.x <= bbox_min.x || max_c.y <= bbox_min.y || max_c.z <= bbox_min.z ||
                    min_c.x >= bbox_max.x || min_c.y >= bbox_max.y || min_c.z >= bbox_max.z ) continue;

                geo_min = fminf(min_c, geo_min);
                geo_max = fmaxf(max_c, geo_max);

                min_c = min_c - bbox_min;
                max_c = max_c - bbox_min;

                int3 min_idx = make_int3(min_c / grid_size);
                int3 max_idx = make_int3(max_c / grid_size);

                min_idx = clamp(min_idx, 0, resolution-1);
                max_idx = clamp(max_idx, 0, resolution-1);

                for (int x=min_idx.x; x<=max_idx.x; x++)
                {
                    for (int y=min_idx.y; y<=max_idx.y; y++)
                    {
                        for (int z=min_idx.z; z<=max_idx.z; z++)
                        {
                            uint32_t n = (x << (log2dim.y + log2dim.z)) | (y << log2dim.z) | z;
                            // printf("x %d y %d z %d n %d\n", x,y,z, n);
                            // uint32_t n = x * resolution.x * resolution.x + y * resolution.y + z;
                            bitmask[n >> 6] |= uint64_t(1) << (n & 63);
                            vis_ptr[n] = true;
                        }
                    }
                }
            }

            if (init_out)
            {
                for (int x=0; x<resolution.x; x++)
                {
                    for (int y=0; y<resolution.y; y++)
                    {
                        for (int z=0; z<resolution.z; z++)
                        {
                            float3 loc = bbox_min + make_float3((float)x,(float)y,(float)z) * grid_size + grid_size / 2.0f;
                            if (loc.x < geo_min.x || loc.y < geo_min.y || loc.z < geo_min.z ||
                                loc.x > geo_max.x || loc.y > geo_max.y || loc.z > geo_max.z)
                            {
                                uint32_t n = (x << (log2dim.y + log2dim.z)) | (y << log2dim.z) | z;
                                bitmask[n >> 6] |= uint64_t(1) << (n & 63);
                                vis_ptr[n] = true;
                            }
                        }
                    }
                }
            }

            printf("init by mesh\n");
        }else{
            for (int i=0; i<size_mask; i++)
            {
                bitmask[i] = ~uint64_t(0);
            }
            for (int i=0; i<resolution.x * resolution.y * resolution.z; i++)
            {
                vis_ptr[i] = true;
            }
            printf("no init\n");
        }


        buffer.create(bitmask, bbox_min, bbox_size, log2dim, size_mask);

        delete [] bitmask;
    }

    void samplePoints(at::Tensor rays_o, at::Tensor rays_d, int num_sample, at::Tensor &z_vals)
    {
        sample_points_cuda(rays_o, rays_d, num_sample, z_vals, buffer);
    }
};


#endif 