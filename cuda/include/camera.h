#ifndef CAMERA_H__
#define CAMERA_H__

enum POSETYPE { WORLD2CAM, CAM2WORLD };

class Intrinsic
{

public:
    float data[9];
    
    __host__ __device__ Intrinsic(){}

    __host__ __device__ Intrinsic(float* _data)
    {   
        #pragma unroll 
        for (int i=0; i<9; i++) data[i] = _data[i];
    }

    __host__ __device__ void create(float* _data)
    {   
        #pragma unroll 
        for (int i=0; i<9; i++) data[i] = _data[i];
    }

    __host__ __device__ Intrinsic inverse()
    {
        float temp[9] = {0};
        temp[0] = 1.0f / data[0];
        temp[4] = 1.0f / data[4];
        temp[2] = -temp[0] * data[2];
        temp[5] = -temp[4] * data[5];
        temp[8] = 1.0f;
        return Intrinsic(temp);
    }

    __host__ __device__ float3 proj(float3 p)
    {
        float x = data[0] * p.x + data[1] * p.y + data[2] * p.z;
        float y = data[3] * p.x + data[4] * p.y + data[5] * p.z;
        float z = data[6] * p.x + data[7] * p.y + data[8] * p.z;
        return make_float3(x,y,z);
    }

    __host__ __device__ void print_data()
    {
        printf("K:\n%f %f %f\n%f %f %f\n%f %f %f\n",
                data[0], data[1], data[2],
                data[3], data[4], data[5],
                data[6], data[7], data[8]);
    }
};

class Extrinsic
{
public:
    float data[12];

    __host__ __device__ Extrinsic(){}
    __host__ __device__ Extrinsic(float* _data)
    {
        #pragma unroll 
        for (int i=0; i<12; i++) data[i] = _data[i];
    }

    __host__ __device__ void create(float* _data)
    {
        #pragma unroll 
        for (int i=0; i<12; i++) data[i] = _data[i];
    }

    __host__ __device__ float3 getTrans()
    {
        return make_float3(data[3], data[7], data[11]);
    }

    __host__ __device__ Extrinsic inverse()
    {
        float temp[12];
        temp[0] = data[0]; temp[1] = data[4]; temp[2] = data[8];
        temp[4] = data[1]; temp[5] = data[5]; temp[6] = data[9];
        temp[8] = data[2]; temp[9] = data[6]; temp[10] = data[10];
        temp[3] = -(temp[0] * data[3] + temp[1] * data[7] + temp[2] * data[11]);
        temp[7] = -(temp[4] * data[3] + temp[5] * data[7] + temp[6] * data[11]);
        temp[11] = -(temp[8] * data[3] + temp[9] * data[7] + temp[10] * data[11]);

        return Extrinsic(temp);
    }
    
    __host__ __device__ float3 rotate(float3 p)
    {
        float x = data[0] * p.x + data[1] * p.y + data[2] * p.z;
        float y = data[4] * p.x + data[5] * p.y + data[6] * p.z;
        float z = data[8] * p.x + data[9] * p.y + data[10] * p.z;
        return make_float3(x,y,z);
    }

    __host__ __device__ float3 trans(float3 p)
    {
        p.x = p.x + data[3];
        p.y = p.y + data[7];
        p.z = p.z + data[11];
        return p;
    }

    __host__ __device__ float3 proj(float3 p)
    {
        return trans(rotate(p));
    }

    __host__ __device__ void print_data()
    {
        printf("E:\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
                data[0], data[1], data[2],
                data[3], data[4], data[5],
                data[6], data[7], data[8],
                data[9], data[10], data[11]);
    }
};

class PinholeCamera
{
public:
    // wrold to camera 
    Intrinsic K;
    Extrinsic E;

    __host__ __device__ PinholeCamera(Intrinsic _K, Extrinsic _E) { K=_K; E=_E; }

    __host__ __device__ PinholeCamera(Intrinsic _K, Extrinsic _E, POSETYPE mode) { 
        K=_K; E=_E; 
        if (mode == CAM2WORLD) inverse_Extrinsic();
    }

    __host__ __device__ PinholeCamera(float* _k, float* _pose, POSETYPE mode=CAM2WORLD){
        K.create(_k); E.create(_pose);
        if (mode == CAM2WORLD) inverse_Extrinsic();
    }

    __host__ __device__ void inverse_Extrinsic(){ E = E.inverse(); }
    __host__ __device__ void inverse_Intrinsic(){ K = K.inverse(); }
    __host__ __device__ void inverse() { inverse_Extrinsic(); inverse_Intrinsic(); }

    __host__ __device__ inline float3 camera_center()
    {
        return E.inverse().getTrans();
    } 

    __host__ __device__ inline float3 pixel2imageplane(float2 uv)
    {
        float3 p = make_float3(uv.x, uv.y, 1.0f);
        return K.inverse().proj(p);
    }

    __host__ __device__ inline float3 imageplace2cam(float3 p, float depth)
    {
        return p * depth;
    }

    __host__ __device__ inline float3 pixel2cam(float2 uv, float depth)
    {
        return imageplace2cam(pixel2imageplane(uv), depth);
    }

    __host__ __device__ inline float3 cam2world(float3 p)
    {
        return E.inverse().proj(p);
    }

    __host__ __device__ inline float3 pixel2world(float2 uv, float depth)
    {
        return cam2world(pixel2cam(uv, depth));
    }

    __host__ __device__ inline float3 world2cam(float3 p)
    {
        return E.proj(p);
    }

    __host__ __device__ inline float3 cam2pixel(float3 p)
    {
        return K.proj(p);
    }

    __host__ __device__ inline float3 world2pixel(float3 p)
    {
        return cam2pixel(world2cam(p));
    }

};

class PinholeCameraManager
{
public:
    Intrinsic* Ks;
    Extrinsic* Es;
    int num_camera;
    POSETYPE mode=WORLD2CAM;
    __host__ __device__ PinholeCameraManager(){}
    __host__ __device__ PinholeCameraManager(Intrinsic *_Ks, Extrinsic* _Es, int _num_camera) 
    { Ks=_Ks; Es=_Es; num_camera=_num_camera;}
    // __host__ __device__ const PinholeCamera& operator[](int i) const { return PinholeCamera(Ks[i], Es[i]);}
    __host__ __device__ const PinholeCamera& operator[](int i) const { 
        Intrinsic K = Ks[i];
        Extrinsic E = Es[i];
        return PinholeCamera(K, E);
    }
};




#endif 