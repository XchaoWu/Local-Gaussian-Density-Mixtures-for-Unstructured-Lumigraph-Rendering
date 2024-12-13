#ifndef GRID__X_H__
#define GRID__X_H__

#include "cutil_math.h"

class VoxelGrid 
{

public:
    uint64_t* data = nullptr;
    int log2dim_x, log2dim_y, log2dim_z;

    __host__ __device__ VoxelGrid(){}

    __host__ __device__ VoxelGrid(uint64_t* grid_data, 
                int log2dim_x, int log2dim_y, int log2dim_z)
    {
        this->data = grid_data;
        this->log2dim_x = log2dim_x;
        this->log2dim_y = log2dim_y;
        this->log2dim_z = log2dim_z;
    }


    // ======= operation ============ //
    __host__ __device__ bool isOn(uint32_t n) { return 0 != (data[n >> 6] & (uint64_t(1) << (n & 63))); }
    __host__ __device__ bool isOff(uint32_t n) { return 0 == (data[n >> 6] & (uint64_t(1) << (n & 63))); }
    __host__ __device__ void setOn(uint32_t n) { data[n >> 6] |= uint64_t(1) << (n & 63); }
    __host__ __device__ void setOff(uint32_t n) { data[n >> 6] &= ~(uint64_t(1) << (n & 63)); }
    // ======= operation ============ //


    __host__ __device__ uint32_t ijk2offset(int i, int j, int k)
    {
        return (i << (log2dim_y + log2dim_z)) | (j << log2dim_z) | k;
    }

    __host__ __device__ uint3 offset2ijk(uint32_t n)
    {
        uint32_t m = n & ((1 << (log2dim_y + log2dim_z)) - 1);
        return make_uint3(n >> (log2dim_y + log2dim_z), m >> log2dim_z, m & ((1 << log2dim_z) - 1));
    }

    __host__ __device__ bool isOn(int i, int j, int k) { return isOn(ijk2offset(i,j,k)); }
    __host__ __device__ bool isOff(int i, int j, int k) { return isOff(ijk2offset(i,j,k)); }
    __host__ __device__ void setOn(int i, int j, int k) { setOn(ijk2offset(i,j,k)); }
    __host__ __device__ void setOff(int i, int j, int k) { setOff(ijk2offset(i,j,k)); }


};


    
#endif 