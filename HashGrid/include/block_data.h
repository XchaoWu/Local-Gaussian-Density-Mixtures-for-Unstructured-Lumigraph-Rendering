#ifndef BLOCKDATA__
#define BLOCKDATA__
#include "cutil_math.h"

struct DataBuffer
{
    uint64_t* data;
    uint64_t loc[4] = {0};

    DataBuffer(){}

    void create(uint64_t* bitmask, float3 block_corner, float3 block_size, uint3 log2dim, uint32_t size_mask)
    {
        // bitmask  uint64_t * size_mask 
        // block_corner 3 x float 
        // block size   3 x float  
        // log2dim      3 x uint32_t 
        cudaMalloc((void**)&data, sizeof(uint64_t)*(size_mask+9));

        cudaMemcpy(data, bitmask, sizeof(uint64_t)*size_mask, cudaMemcpyHostToDevice);
        cudaMemcpy(data+size_mask, &block_corner.x, sizeof(float)*3, cudaMemcpyHostToDevice);
        cudaMemcpy(data+size_mask+3, &block_size.x, sizeof(float)*3, cudaMemcpyHostToDevice);
        cudaMemcpy(data+size_mask+6, &log2dim.x, sizeof(uint32_t)*3, cudaMemcpyHostToDevice);

        loc[0] = 0;
        loc[1] = loc[0] + size_mask;
        loc[2] = loc[1] + 3;
        loc[3] = loc[2] + 3;

        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }else{
            printf("create databuffer successfully!\n");
        }
    }

    void free(){
        cudaFree(data);
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }else{
            printf("delete databuffer successfully!\n");
        }
    }

    __device__ 
    uint64_t* getBitmask()
    {
        return data;
    }
    __device__
    float3 getBlockCorner()
    {
        float* ptr = (float*)(data + loc[1]);
        return make_float3(ptr[0],ptr[1],ptr[2]);
    }
    __device__ 
    float3 getBlockSize()
    {
        float* ptr = (float*)(data + loc[2]);
        return make_float3(ptr[0],ptr[1],ptr[2]);
    }
    __device__ 
    uint3 getLog2Dim()
    {
        uint32_t* ptr = (uint32_t*)(data + loc[3]);
        return make_uint3(ptr[0],ptr[1],ptr[2]);
    }

};


#endif 