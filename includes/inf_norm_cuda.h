/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * modified by Phillip Weinberg to produce the infinite norm of a vector. 
 */

/*
    Parallel reduction kernels
*/

#ifndef __INF_NORM_CUDA_H__
#define __INF_NORM_CUDA_H__

#include <stdio.h>
#include "error.h"
#include <cuda.h>

#ifndef __complex_types__
#define __complex_types__

#include <thrust/complex.h>
#include <cuComplex.h>

typedef thrust::complex<double> cdouble;
typedef thrust::complex<float> cfloat;

#endif

namespace inf_norm_math {
__device__ inline float abs(float val){
    return fabsf(val);
}

__device__ inline double abs(double val){
    return fabs(val);
}

__device__ inline float abs(thrust::complex<float> val){
    return thrust::abs(val);
}

__device__ inline double abs(thrust::complex<double> val){
    return thrust::abs(val);
}

__device__ inline double max(double a,double b){
    return fmax(a,b);
}

__device__ inline float max(float a,float b){
    return fmaxf(a,b);
}

}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <typename realT, typename T, unsigned int blockSize, bool nIsPow2>
__global__ void
inf_norm_kernel(T *g_idata, realT *g_odata, unsigned int n)
{
    realT *sdata = SharedMemory<realT>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    realT myMax = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        myMax = inf_norm_math::abs(g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            myMax = inf_norm_math::max(myMax,inf_norm_math::abs(g_idata[i+blockSize]));

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = myMax;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 256]);
        // sdata[tid] = myMax = myMax + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 128]);
        // sdata[tid] = myMax = myMax + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 64]);
        // sdata[tid] = myMax = myMax + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) myMax = inf_norm_math::max(myMax,sdata[tid + 32]);
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            // myMax = inf_norm_math::max(myMax,__shfl_down(myMax, offset));
            myMax = inf_norm_math::max(myMax,__shfl_down_sync(0xFFFFFFFF, myMax, offset, 32));
        }
        /*
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) myMax += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            myMax += __shfl_down(myMax, offset);
        }
        */
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 32]);
        // sdata[tid] = myMax = myMax + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 16]);
        // sdata[tid] = myMax = myMax + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 8]);
        // sdata[tid] = myMax = myMax + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 4]);
        // sdata[tid] = myMax = myMax + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 2]);
        // sdata[tid] = myMax = myMax + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = myMax = inf_norm_math::max(myMax,sdata[tid + 1]);
        // sdata[tid] = myMax = myMax + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = myMax;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#define __MAX_THREADS_INF_NORM 512

template <typename I,typename realT,typename T>
void inf_norm_case(const I size,const unsigned int blocks,const unsigned int threads, T *d_idata, realT *d_odata, cudaStream_t stream=(cudaStream_t)0)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    
    bool ispow2 = !((~(~0U>>1)|size)&size -1);

    if(ispow2){
        switch(threads){
            case 512:
                inf_norm_kernel<realT, T, 512, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case 256:
                inf_norm_kernel<realT, T, 256, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case 128:
                inf_norm_kernel<realT, T, 128, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case  64:
                inf_norm_kernel<realT, T,  64, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case  32:
                inf_norm_kernel<realT, T,  32, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case  16:
                inf_norm_kernel<realT, T,  16, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   8:
                inf_norm_kernel<realT, T,   8, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   4:
                inf_norm_kernel<realT, T,   4, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   2:
                inf_norm_kernel<realT, T,   2, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   1:
                inf_norm_kernel<realT, T,   1, true ><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
        }
    }
    else{
        switch(threads){
            case 512:
                inf_norm_kernel<realT, T, 512, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case 256:
                inf_norm_kernel<realT, T, 256, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case 128:
                inf_norm_kernel<realT, T, 128, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case  64:
                inf_norm_kernel<realT, T,  64, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case  32:
                inf_norm_kernel<realT, T,  32, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case  16:
                inf_norm_kernel<realT, T,  16, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   8:
                inf_norm_kernel<realT, T,   8, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   4:
                inf_norm_kernel<realT, T,   4, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   2:
                inf_norm_kernel<realT, T,   2, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
            case   1:
                inf_norm_kernel<realT, T,   1, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, (unsigned int)size);
                break;
        }
    }


}


unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}




//////////////////////////////////////////
// Wrapper function for kernel launches //
//////////////////////////////////////////
template <typename I,typename realT,typename T>
void inf_norm(const I size, T *d_idata, realT *d_odata,realT *res_host, cudaStream_t stream=(cudaStream_t)0)
{
    unsigned int threads = (size < 2*__MAX_THREADS_INF_NORM) ? nextPow2((size + 1)/ 2) : __MAX_THREADS_INF_NORM;
    unsigned int blocks  = (size + (threads * 2 - 1)) / (threads * 2);

    inf_norm_case<I,realT,T>(size,blocks,threads,d_idata,d_odata,stream);

    // sum partial block sums on GPU
    I s=blocks;
    while (s > 1)
    {
        threads = (size < 2*__MAX_THREADS_INF_NORM) ? nextPow2((size + 1)/ 2) : __MAX_THREADS_INF_NORM;
        blocks  = (s + (threads * 2 - 1)) / (threads * 2);

        inf_norm_case<I,realT,realT>(s,blocks,threads,d_odata,d_odata,stream);
        s = (s + (threads*2-1)) / (threads*2);

    }
    cudaMemcpyAsync(res_host,d_odata,sizeof(realT),cudaMemcpyDeviceToHost,stream);
}



#endif // #ifndef _REDUCE_KERNEL_H_