#ifndef DDA_OVERLAP_H__
#define DDA_OVERLAP_H__

// FIX do not change 
#define SPLIT_NUM 10  // overlap ratio = 0.1 


struct DDAOverlap{

    // not index space 

    uint32_t SIDE;
    float tile_size;

    uint32_t BLOCK_SIDE;

    int3 tstep;
    float3 tMax;
    float3 tDelta;
    int3 current_tile; // could be nodes and voxels
    int3 mask;
    float2 t;

    __device__ void init(float3 origin, float3 direction, float2 t_start, 
                         float block_size, uint32_t block_side)
    {
        BLOCK_SIDE = block_side;
        SIDE = (SPLIT_NUM - 1) * BLOCK_SIDE + 1;
        tile_size = block_size / SPLIT_NUM;

        origin = origin + t_start.x * direction;
        current_tile = make_int3(origin / tile_size);
        current_tile = clamp(current_tile, 0, (int)SIDE - 1);
        tstep = signf(direction);

        float3 next_boundary = make_float3(current_tile + tstep) * tile_size;

        if (tstep.x < 0) next_boundary.x += tile_size;
        if (tstep.y < 0) next_boundary.y += tile_size;
        if (tstep.z < 0) next_boundary.z += tile_size;
        t = t_start; // init 
        tMax = fmaxf(safe_divide(next_boundary-origin, direction), 0.0f) + t.x;
        tDelta = fabs(safe_divide(make_float3(tile_size), direction));
    }

    __device__ void next()
    {
		mask.x = int((tMax.x < tMax.y) & (tMax.x <= tMax.z));
		mask.y = int((tMax.y < tMax.z) & (tMax.y <= tMax.x));
        mask.z = !(mask.x | mask.y);
		// mask.z = int((tMax.z < tMax.x) & (tMax.z <= tMax.y));
        t.y = mask.x ? tMax.x : (mask.y ? tMax.y : tMax.z);
    }

    __device__ void step()
    {
        t.x = t.y;
        tMax += make_float3(mask) * tDelta;
        current_tile += mask * tstep;
    }


    __device__ bool terminate()
    {
        return current_tile.x < 0 || current_tile.y < 0 || current_tile.z < 0 || 
               current_tile.x >= SIDE || current_tile.y >= SIDE || current_tile.z >= SIDE ||
               (tMax.x <= 0 && tMax.y <= 0 && tMax.z <= 0);
    }

    __device__ int getBlockIdx(int* blockIdx)
    {

        /*
        0,1,2, ..., 9   -> 0
        9,10,11,...,18  -> 1
        18,19,20,...,27 -> 2
        */
        int3 xyz = current_tile / (SPLIT_NUM - 1);
        int3 mod = current_tile - xyz * (SPLIT_NUM - 1);
        
        // 首先可以得到的
        int count = 0;
        // printf("xyz %d %d %d\n", xyz.x, xyz.y, xyz.z);
        // 边界 block 情况
        if(xyz.x < BLOCK_SIDE && xyz.y < BLOCK_SIDE && xyz.z < BLOCK_SIDE)
        {
            int3 loc = xyz;
            blockIdx[count++] = loc.x * BLOCK_SIDE * BLOCK_SIDE + loc.y * BLOCK_SIDE + loc.z;
        }

        if (mod.x == 0 && mod.z != 0)
        {
            xyz.x -= 1;
            if (xyz.x >= 0)
            {
                int3 loc = xyz;
                blockIdx[count++] = loc.x * BLOCK_SIDE * BLOCK_SIDE + loc.y * BLOCK_SIDE + loc.z;
            }
        }else if (mod.x != 0 && mod.z == 0)
        {
            xyz.z -= 1;
            if (xyz.z >= 0)
            {
                int3 loc = xyz;
                blockIdx[count++] = loc.x * BLOCK_SIDE * BLOCK_SIDE + loc.y * BLOCK_SIDE + loc.z;
            }
        }else if (mod.x == 0 && mod.z == 0)
        {
            // 4 blocks 
            if (xyz.x >= 1)
            {
                int3 loc = make_int3(xyz.x-1, xyz.y, xyz.z);
                blockIdx[count++] = loc.x * BLOCK_SIDE * BLOCK_SIDE + loc.y * BLOCK_SIDE + loc.z;
            }

            if (xyz.z >= 1)
            {
                int3 loc = make_int3(xyz.x, xyz.y, xyz.z-1);
                blockIdx[count++] = loc.x * BLOCK_SIDE * BLOCK_SIDE + loc.y * BLOCK_SIDE + loc.z;
            }

            if (xyz.x >= 1 && xyz.z >= 1)
            {
                int3 loc = make_int3(xyz.x-1, xyz.y, xyz.z-1);
                blockIdx[count++] = loc.x * BLOCK_SIDE * BLOCK_SIDE + loc.y * BLOCK_SIDE + loc.z;
            }

        }

        return count;
    }


};

#endif 