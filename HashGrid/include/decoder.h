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
    float* params;
    // half* params_globalmem; // params in global mem 

public:
    
    __hostdev__ Decoder(){}
    __hostdev__ ~Decoder(){}

    __device__ Decoder(float* _params)
    {
        params = _params;
    }

    __device__ void SH_encoder_deg3(float3 direction, float* layer)
    {
        float x = direction.x;
        float y = direction.y;
        float z = direction.z;

        uint32_t index = 0;

        // 1
        layer[index++] = C0;
        // 3
        layer[index++] = C1 * y;
        layer[index++] = C1 * z;
        layer[index++] = C1 * x;

        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, yz = y*z, xz = x*z;
        // 5
        layer[index++] = C20 * xy;
        layer[index++] = C21 * yz;
        layer[index++] = C22 * (2.0 * zz - xx - yy);
        layer[index++] = C23 * xz;
        layer[index++] = C24 * (xx - yy);

        // 7 
        layer[index++] = C30 * y * (3 * xx - yy);
        layer[index++] = C31 * xy * z;
        layer[index++] = C32 * y * (4 * zz - xx - yy);
        layer[index++] = C33 * z * (2 * zz - 3 * xx - 3 * yy);
        layer[index++] = C34 * x * (4 * zz - xx - yy);
        layer[index++] = C35 * z * (xx - yy);
        layer[index++] = C36 * x * (xx - 3 * yy);

    }

    template<uint32_t SIZE>
    __inline__ __device__  
    void gaussian_act(float* x)
    {
        #pragma unroll
        for (int i=0; i<SIZE; i++)
        {
            x[i] = expf(x[i]*x[i] / -0.02f);
        }
    }


    __inline__ __device__
    float softplus(float x)
    {
        return logf(1.0f + expf(x));
    }

    __inline__ __device__ 
    float sigmoid(float x)
    {
        return 1.0f / (1.0f + expf(-1.0f * x));
    }

    __inline__ __device__
    float3 sigmoid(float* x)
    {
        return make_float3( sigmoid(x[0]), sigmoid(x[1]), sigmoid(x[2]));
    }

    template <uint32_t INPUT_DIM, uint32_t OUT_DIM, uint32_t OFFSET>
    __inline__ __device__ 
    void Linear(int &param_index, float* layer)
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
                layer[j] += layer[i] * params[param_index++];
            }
        }
    }

    __device__ void inference(float* inputs, float3 direction, float3 &diffuse, float3 &specular, float &sigma)
    {
        
        float layer[HIDEN_WIDTH * 2]; 

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
        sigma = softplus(layer[HIDEN_WIDTH]);  
        // sigma = softplus(layer[HIDEN_WIDTH]);

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

        Linear<32+16, HIDEN_WIDTH, HIDEN_WIDTH>(param_index, layer);
        gaussian_act<HIDEN_WIDTH>(layer+HIDEN_WIDTH);

        Linear<HIDEN_WIDTH, HIDEN_WIDTH, 0>(param_index, layer);
        gaussian_act<HIDEN_WIDTH>(layer);

        Linear<HIDEN_WIDTH, 3, HIDEN_WIDTH>(param_index, layer);
        specular = tint * sigmoid(layer+HIDEN_WIDTH);
    }

};

#endif 