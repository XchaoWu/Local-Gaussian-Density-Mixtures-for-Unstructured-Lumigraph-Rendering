#ifndef DDA_H__
#define DDA_H__

#include "cutil_math.h"

template<u_int32_t LOG2DIM, uint32_t CLOG2DIM>
struct DDASatate{

    static constexpr uint32_t SIDE = 1u << LOG2DIM;
    static constexpr uint32_t TILE_SIZE = 1u << CLOG2DIM;

    int3 tstep;
    float3 tMax;
    float3 tDelta;
    int3 current_tile; // could be nodes and voxels
    int3 mask;
    float2 t;


    __device__ void init(float3 origin, float3 direction, float2 t_start)
    {
        // printf("TILE_SIZE %d\n", TILE_SIZE);

        // [TODO] may should clamp origin ?
        origin = origin + t_start.x * direction;
        current_tile = make_int3(origin / TILE_SIZE);
        current_tile = clamp(current_tile, 0, (int)SIDE-1);
        tstep = signf(direction);

        float3 next_boundary = make_float3(current_tile + tstep) * TILE_SIZE;

        if (tstep.x < 0) next_boundary.x += TILE_SIZE;
        if (tstep.y < 0) next_boundary.y += TILE_SIZE;
        if (tstep.z < 0) next_boundary.z += TILE_SIZE;
        t = t_start; // init 
        tMax = fmaxf(safe_divide(next_boundary-origin, direction), 0.0f) + t.x;
        tDelta = fabs(safe_divide(make_float3(TILE_SIZE), direction));
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

    __device__ uint3 indexSpaceLoc()
    {

        return make_uint3(current_tile * TILE_SIZE);
    }
};


struct DDASatateV2{

    uint32_t SIDE;
    uint32_t TILE_SIZE;

    int3 tstep;
    float3 tMax;
    float3 tDelta;
    int3 current_tile; // could be nodes and voxels
    int3 mask;
    float2 t;


    __device__ void init(float3 origin, float3 direction, float2 t_start,
                         uint32_t _SIDE, uint32_t _TILE_SIZE)
    {
        SIDE = _SIDE;
        TILE_SIZE = _TILE_SIZE;
        // printf("TILE_SIZE %d\n", TILE_SIZE);

        // [TODO] may should clamp origin ?
        origin = origin + t_start.x * direction;
        current_tile = make_int3(origin / TILE_SIZE);
        current_tile = clamp(current_tile, 0, (int)SIDE-1);
        tstep = signf(direction);

        float3 next_boundary = make_float3(current_tile + tstep) * TILE_SIZE;

        if (tstep.x < 0) next_boundary.x += TILE_SIZE;
        if (tstep.y < 0) next_boundary.y += TILE_SIZE;
        if (tstep.z < 0) next_boundary.z += TILE_SIZE;
        t = t_start; // init 
        tMax = fmaxf(safe_divide(next_boundary-origin, direction), 0.0f) + t.x;
        tDelta = fabs(safe_divide(make_float3(TILE_SIZE), direction));
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

    __device__ uint3 indexSpaceLoc()
    {

        return make_uint3(current_tile * TILE_SIZE);
    }
};



struct DDASatateScene{


    int3 tstep;
    float3 tMax;
    float3 tDelta;
    int3 current_tile; // could be nodes and voxels
    int3 mask;
    float2 t;
    uint32_t SIDE;
    float TILE_SIZE;


    __device__ void init(float3 origin, float3 direction, float2 t_start, 
                        uint32_t _SIDE, float _TILE_SIZE)
    {
        
        SIDE = _SIDE; TILE_SIZE = _TILE_SIZE;
        // [TODO] may should clamp origin ?
        origin = origin + t_start.x * direction;
        current_tile = make_int3(origin / TILE_SIZE);
        current_tile = clamp(current_tile, 0, (int)SIDE-1);


        tstep = signf(direction);

        float3 next_boundary = make_float3(current_tile + tstep) * TILE_SIZE;

        if (tstep.x < 0) next_boundary.x += TILE_SIZE;
        if (tstep.y < 0) next_boundary.y += TILE_SIZE;
        if (tstep.z < 0) next_boundary.z += TILE_SIZE;

        t = t_start; // init 

        tMax = fmaxf(safe_divide(next_boundary-origin, direction), 0.0f) + t.x;
        tDelta = fabs(safe_divide(make_float3(TILE_SIZE), direction));

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

};




struct DDASatateScene_v2{


    int3 tstep;
    float3 tMax;
    float3 tDelta;
    int3 current_tile; // could be nodes and voxels
    int3 mask;
    float2 t;
    int3 SIDE;
    float3 TILE_SIZE;


    __device__ void init(float3 origin, float3 direction, float2 t_start, 
                        int3 _SIDE, float3 _TILE_SIZE)
    {
        
        SIDE = _SIDE; TILE_SIZE = _TILE_SIZE;
        // [TODO] may should clamp origin ?
        origin = origin + t_start.x * direction;
        current_tile = make_int3(origin / TILE_SIZE);
        current_tile = clamp(current_tile, 0, SIDE-1);


        tstep = signf(direction);

        float3 next_boundary = make_float3(current_tile + tstep) * TILE_SIZE;

        if (tstep.x < 0) next_boundary.x += TILE_SIZE.x;
        if (tstep.y < 0) next_boundary.y += TILE_SIZE.y;
        if (tstep.z < 0) next_boundary.z += TILE_SIZE.z;

        t = t_start; // init 

        tMax = fmaxf(safe_divide(next_boundary-origin, direction), 0.0f) + t.x;
        tDelta = fabs(safe_divide(TILE_SIZE, direction));

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
               current_tile.x >= SIDE.x || current_tile.y >= SIDE.y || current_tile.z >= SIDE.z ||
               (tMax.x <= 0 && tMax.y <= 0 && tMax.z <= 0);
    }

};

#endif 