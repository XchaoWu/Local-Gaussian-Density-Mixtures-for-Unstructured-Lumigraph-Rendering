#ifndef DECODER_H__
#define DECODER_H__

#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/ATen.h>
#include <cassert>
#include <cuda.h>
#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include "cutil_math.h"
#include "cuda_utils.h"

#define PARAMSIZE 13994 
#define HIDEN_WIDTH 64 


#define C0 0.28209479177387814f
#define C1 0.4886025119029199f
#define C20 1.0925484305920792f
#define C21 -1.0925484305920792f
#define C22 0.31539156525252005f
#define C23 -1.0925484305920792f
#define C24 0.5462742152960396f
#define C30 -0.5900435899266435f
#define C31 2.890611442640554f
#define C32 -0.4570457994644658f
#define C33 0.3731763325901154f
#define C34 -0.4570457994644658f
#define C35 1.445305721320277f
#define C36 -0.5900435899266435f
#define C40 2.5033429417967046f
#define C41 -1.7701307697799304f
#define C42 0.9461746957575601f
#define C43 -0.6690465435572892f
#define C44 0.10578554691520431f
#define C45 -0.6690465435572892f
#define C46 0.47308734787878004f
#define C47 -1.7701307697799304f
#define C48 0.6258357354491761f

/*
Mem layout 

Spatial MLP
64 64 x 24 
64 64 x 64 

sigma layer 
1  1 x 32
diffuse layer 
3  3 x 32
tint layer 
3  3 x 32

Directional MLP 
64 64 x 35  
64 64 x 64 
3  3 x 64

*/

class Decoder
{
    half* params;
    // half* params_globalmem; // params in global mem 

public:
    
    __hostdev__ Decoder(){}
    __hostdev__ ~Decoder(){}

    __device__ Decoder(half* _params)
    {
        params = _params;
    }

    __device__ void SH_encoder_deg3(float3 direction, half* layer)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
        half x = __float2half(direction.x);
        half y = __float2half(direction.y);
        half z = __float2half(direction.z);

        uint32_t index = 0;

        // 1
        layer[index++] = __float2half(C0);
        // 3
        layer[index++] = __hmul(__float2half(C1), y);
        layer[index++] = __hmul(__float2half(C1), z);
        layer[index++] = __hmul(__float2half(C1), x);

        half xx = __hmul(x,x); half yy = __hmul(y,y); half zz = __hmul(z,z);
        half xy = __hmul(x,y); half yz = __hmul(y,z); half xz = __hmul(x,z);
        // 5
        layer[index++] = __hmul(__float2half(C20), xy);
        layer[index++] = __hmul(__float2half(C21), yz);
        layer[index++] = __hmul(__float2half(C22), __hsub(__hsub(__hmul(__float2half(2.0f), zz),xx),yy));
        layer[index++] = __hmul(__float2half(C23), xz);
        layer[index++] = __hmul(__float2half(C24), __hsub(xx,yy));

        // 7 
        layer[index++] = __hmul(__float2half(C30), __hmul(y, __hsub(__hmul(__float2half(3.0f), xx), yy)));
        layer[index++] = __hmul(__float2half(C31), __hmul(xy, z));
        layer[index++] = __hmul(__float2half(C32), __hmul(y, __hsub(__hsub(__hmul(__float2half(4.0f), zz),xx),yy)));
        layer[index++] = __hmul(__float2half(C33), __hmul(z, __hsub(__hsub( __hmul(__float2half(2.0f), zz), __hmul(__float2half(3.0f), xx)), __hmul(__float2half(3.0f),yy) )));
        layer[index++] = __hmul(__float2half(C34), __hmul(x, __hsub(__hsub(__hmul(__float2half(4.0f), zz),xx),yy)));
        layer[index++] = __hmul(__float2half(C35), __hmul(z, __hsub(xx, yy)));
        layer[index++] = __hmul(__float2half(C36), __hmul(x, __hsub(xx, __hmul(__float2half(3.0f), yy))));
#else 
            
#endif 
    }

    template<uint32_t SIZE>
    __inline__ __device__  
    void gaussian_act(half* x)
    {
        half sigma = __float2half(0.1f);
        #pragma unroll
        for (int i=0; i<SIZE; i++)
        {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
            x[i] = hexp( __hdiv( __hmul(x[i], x[i]) , __float2half(-0.02f)) );
#else 
            
#endif 
        }
    }


    __inline__ __device__
    float softplus(float x)
    {
        return logf(1.0f + expf(x));
    }

    __inline__ __device__ 
    float sigmoid(half x)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
        return __half2float(__hdiv( __float2half(1.0f),  __hadd( __float2half(1.0f), hexp(  __hmul(__float2half(-1.0f), x) ) )));
#else 
#endif 
    }

    __inline__ __device__
    float3 sigmoid(half* x)
    {
        return make_float3( sigmoid(x[0]), sigmoid(x[1]), sigmoid(x[2]));
    }

    template <uint32_t INPUT_DIM, uint32_t OUT_DIM, uint32_t OFFSET>
    __inline__ __device__ 
    void Linear(int &param_index, half* layer)
    {
        #pragma unroll
        for (int i=OFFSET; i<OFFSET+OUT_DIM; i++)
        {
            layer[i] = params[param_index++];
        }

        #pragma unroll
        for (int i=HIDEN_WIDTH-OFFSET; i<HIDEN_WIDTH-OFFSET+INPUT_DIM; i++)
        {
            for (int j=OFFSET; j<OFFSET+OUT_DIM; j++)
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
                layer[j] = __hfma(layer[i], params[param_index++], layer[j]);
#else 
#endif 
            }
        }
    }

    __device__ void inference(half* inputs, float3 direction, float3 &diffuse, float3 &specular, float &sigma)
    {
        
        half layer[HIDEN_WIDTH * 2]; 

        int param_index = 0;

        #pragma unroll
        for (int i=0; i<32; i++)
        {
            layer[i] = inputs[i];
        }

        // input, output, offset
        Linear<32, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, layer);
        gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);
        Linear<HIDEN_WIDTH, HIDEN_WIDTH, 0>(param_index, layer);

        // sigma 
        Linear<HIDEN_WIDTH / 2, 1, HIDEN_WIDTH>(param_index, layer);
        sigma = softplus(__half2float(layer[HIDEN_WIDTH]));   

        // c_d
        Linear<HIDEN_WIDTH / 2, 3, HIDEN_WIDTH>(param_index, layer);
        diffuse = sigmoid(layer+HIDEN_WIDTH);

        // tint 
        Linear<HIDEN_WIDTH / 2, 3, HIDEN_WIDTH>(param_index, layer);
        float3 tint = sigmoid(layer+HIDEN_WIDTH);
        
        // Directional MLP 
        direction = normalize(direction);
        #pragma unroll
        for (int i=0; i<HIDEN_WIDTH/2; i++)
        {
            layer[i] = layer[i+HIDEN_WIDTH/2];
        }

        SH_encoder_deg3(direction, layer+32);
        // layer[32] = __float2half(direction.x);
        // layer[33] = __float2half(direction.y);
        // layer[34] = __float2half(direction.z);

        Linear<32+16, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, layer);
        gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);

        Linear<HIDEN_WIDTH, HIDEN_WIDTH, 0>(param_index, layer);
        gaussian_act<HIDEN_WIDTH>(layer);

        Linear<HIDEN_WIDTH, 3, HIDEN_WIDTH>(param_index, layer);
        specular = tint * sigmoid(layer+HIDEN_WIDTH);
    }

};

#endif 