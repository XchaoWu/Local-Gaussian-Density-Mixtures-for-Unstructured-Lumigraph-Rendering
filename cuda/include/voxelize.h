#ifndef VOXELIZE___ 
#define VOXELIZE___ 

#include "plyIO.h"
#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>
#define INF 100000000.0f
void voxelize_mesh(at::Tensor _log2dim, 
                   at::Tensor block_corner, 
                   at::Tensor block_size, 
                   std::string model_path, 
                   at::Tensor &vis,
                   bool init_out)
{
    uint3 log2dim = make_uint3(_log2dim.contiguous().data_ptr<int>()[0],
                            _log2dim.contiguous().data_ptr<int>()[1],
                            _log2dim.contiguous().data_ptr<int>()[2]);
    float3 bbox_size = make_float3(block_size.contiguous().data_ptr<float>()[0],
                                block_size.contiguous().data_ptr<float>()[1],
                                block_size.contiguous().data_ptr<float>()[2]);
    int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);

    float3 grid_size = bbox_size / make_float3((float)resolution.x, (float)resolution.y, (float)resolution.z); 
    float3 bbox_min = make_float3(block_corner.contiguous().data_ptr<float>()[0],
                               block_corner.contiguous().data_ptr<float>()[1],
                               block_corner.contiguous().data_ptr<float>()[2]);
    float3 bbox_max = bbox_min + bbox_size;
    bool* vis_ptr = vis.contiguous().data_ptr<bool>();
    // bool* out_ptr = outside.contiguous().data_ptr<bool>();
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
                            vis_ptr[n] = true;
                            // out_ptr[n] = true;
                        }
                    }
                }
            }
        }

        printf("init by mesh\n");
    }else{
        for (int i=0; i<resolution.x * resolution.y * resolution.z; i++)
        {
            vis_ptr[i] = true;
        }
        printf("no init\n");
    }

}

#endif 