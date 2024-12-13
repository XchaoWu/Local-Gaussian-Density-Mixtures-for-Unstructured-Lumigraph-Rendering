#ifndef __BUILD_BLOCKS__H__
#define __BUILD_BLOCKS__H__

#include <iostream>
#include <fstream>
#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include "plyIO.h"
#include <cnpy.h>
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
#include "camera.h"
namespace py = pybind11;


void compute_relate_matrix(
    at::Tensor &related_matrix, // num_camera x num_tile
    PinholeCameraManager cameras, 
    at::Tensor depths,
    float* block_corner, float block_size, 
    float tile_size, int tile_dim, int* IndexMap, 
    int height, int width);


class BlockBuilder
{
    std::vector<float3> vertices;
    std::vector<int3> faces;
    std::vector<int> block_list; // save dense idx 
    float3* block_corners;
    std::vector<cnpy::NpyArray> depths;
    PinholeCameraManager cameras;
    int height, width;
    float block_size;

public:
    BlockBuilder(){}
    ~BlockBuilder(){}

    void load_mesh(std::string path)
    {
        std::vector<float2> uv_array;
        read_plyFile(path, vertices, faces, uv_array);
    }
    

    void load_depths(std::vector<std::string> path_list)
    {
        for (int i=0; i<path_list.size(); i++)
        {
            depths.emplace_back(cnpy::npy_load(path_list[i]));
        }
    }

    void load_camera(at::Tensor ks, at::Tensor rts, int H, int W)
    {
        int num_camera = ks.size(0);

        cameras = PinholeCameraManager((Intrinsic*)ks.contiguous().data_ptr<float>(), 
                                       (Extrinsic*)rts.contiguous().data_ptr<float>(), 
                                       num_camera);
        
        height = H; width = W;
    }


    at::Tensor view_selection_for_single_block(int bidx, int TILE_DIM, at::Tensor depths,
                                               at::Tensor& tile_centers, at::Tensor &corner)
    {
        float3 block_corner = block_corners[bidx];
        printf("block_corner %f %f %f\n", block_corner.x, block_corner.y, block_corner.z);
        corner[0] = block_corner.x;
        corner[1] = block_corner.y;
        corner[2] = block_corner.z;

        int tile_side = 1u << TILE_DIM;
        int total_num = 1u << (3 * TILE_DIM);
        float tile_size = block_size / (float)tile_side;

        bool* occupied = new bool[total_num];
        for (int i=0; i<total_num; i++)
        {
            occupied[i] = false;
        }
        
        float3* tile_centers_ptr = (float3*)tile_centers.contiguous().data_ptr<float>();
        for (int i=0; i<faces.size(); i++)
        {
            int3 vidx = faces[i];
            float3 A = vertices[vidx.x];
            float3 B = vertices[vidx.y];
            float3 C = vertices[vidx.z];

            float3 tri_center = (A + B + C) / 3.0f;

            int3 loc = make_int3((tri_center - block_corner) / tile_size);
            if (loc.x >= 0 && loc.x < tile_side && 
                loc.y >= 0 && loc.y < tile_side && 
                loc.z >= 0 && loc.z < tile_side)
            {
                int n = (loc.x << (2 * TILE_DIM)) | (loc.y << TILE_DIM) | loc.z;
                occupied[n] = true;

                // output tile for this block 
                float3 tile_center = block_corner + make_float3(loc) * tile_size + tile_size / 2.0f;
                tile_centers_ptr[n] = tile_center;
            }
        }

        int* IndexMap = new int[total_num];
        int count = 0;
        for (int i=0; i<total_num; i++)
        {
            if(occupied[i])
            {
                IndexMap[i] = count;
                count++;
            }else{
                IndexMap[i] = -1;
            }
        }
        delete [] occupied;



        int* _IndexMap;
        cudaMalloc((void**)&_IndexMap, sizeof(int)*total_num);
        cudaMemcpy(_IndexMap, IndexMap, sizeof(int)*total_num, cudaMemcpyHostToDevice);
        float* _block_corner;
        cudaMalloc((void**)&_block_corner, sizeof(float3));
        cudaMemcpy(_block_corner, &block_corner.x, sizeof(float3), cudaMemcpyHostToDevice);

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        at::Tensor related_matrix = torch::full({cameras.num_camera, count}, 0, options);


        compute_relate_matrix(related_matrix, cameras, depths, _block_corner, 
                              block_size, tile_size, TILE_DIM, _IndexMap, height, width);

        cudaFree(_IndexMap);
        cudaFree(_block_corner);

        return related_matrix;
    }

    at::Tensor init_blocks(uint32_t BLOCK_DIM, float _block_size, 
                           at::Tensor &scene_min_corner_arr,
                           float overlap_ratio=0.1, 
                           float offset_x=0, float offset_y=0, float offset_z=0)
    {

        block_size =  _block_size;
        float3 scene_min_corner = make_float3(1000000,1000000,1000000);
        float3 scene_max_corner = make_float3(-1000000,-1000000,-1000000);

        for (int i=0; i<vertices.size(); i++)
        {
            scene_min_corner = fminf(vertices[i], scene_min_corner);
            scene_max_corner = fmaxf(vertices[i], scene_max_corner);
        }
        scene_min_corner = scene_min_corner + make_float3(offset_x, offset_y, offset_z);

        int block_side = 1u << BLOCK_DIM; 
        int total_num = 1u << (3 * BLOCK_DIM);

        bool* occupied = new bool[total_num];
        block_corners = new float3[total_num];
        for (int i=0; i<total_num; i++)
        {
            occupied[i] = false;
        }
        for (int x=0; x<block_side; x++)
        {
            for (int y=0; y<block_side; y++)
            {
                for (int z=0; z<block_side; z++)
                {
                    int n = (x << (2*BLOCK_DIM)) | (y << BLOCK_DIM) | z; 
                    block_corners[n] = scene_min_corner + \
                            make_float3((float)x,(float)y,(float)z) * block_size * (1.0f - 2.0f * overlap_ratio);
                }
            }
        }

        for (int i=0; i<faces.size(); i++)
        {
            int3 vidx = faces[i];
            float3 A = vertices[vidx.x];
            float3 B = vertices[vidx.y];
            float3 C = vertices[vidx.z];

            float3 tri_center = (A + B + C) / 3.0f;

            for (int j=0; j<total_num; j++)
            {
                if (occupied[j]) continue;

                float3 block_corner = block_corners[j];
                float3 dis = tri_center - block_corner;

                if (dis.x >= 0 && dis.x < block_size && dis.y >= 0 && dis.y < block_size && dis.z >= 0 && dis.z < block_size)
                {
                    occupied[j] = true;
                }
            }
        }

        for (int i=0; i<total_num; i++)
        {
            if (occupied[i])
            {
                block_list.emplace_back(i);
            }
        }

        delete [] occupied;

        auto options = torch::TensorOptions().dtype(torch::kInt32);
        at::Tensor out = torch::full({(int)block_list.size()}, 0, options);
        int* out_ptr = out.contiguous().data_ptr<int>();

        std::cout << "occupied block index:\n";
        for (int i=0; i<block_list.size(); i++)
        {
            out_ptr[i] = block_list[i];
            std::cout << block_list[i] << " ";
        }
        std::cout << std::endl;

        scene_min_corner_arr[0] = scene_min_corner.x;
        scene_min_corner_arr[1] = scene_min_corner.y;
        scene_min_corner_arr[2] = scene_min_corner.z;

        return out;
    }

};



#endif 