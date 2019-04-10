#ifndef __EXPM_MULTIPLY_GPU_SRC__
#define __EXPM_MULTIPLY_GPU_SRC__


#include <cuda_runtime.h>
#include <thrust/complex.h>
#include "inf_norm_cuda.h"
#include "csrmv.h"
#include "error.h"


#ifndef __complex_types__
#define __complex_types__

#include <thrust/complex.h>
#include <cuComplex.h>

typedef thrust::complex<double> cdouble;
typedef thrust::complex<float> cfloat;

#endif



template<typename I,typename realT,typename T>
class expm_multiply_gpu
{
public:

    // expm_multiply data
    const int s;
    const int m_star;
    const realT tol;
    const T eta;
    const I n_rows;

    unsigned int maxThreads,minBlocks;
    cudaStream_t s1,s2;


    T* F;
    T* B1;
    T* B2;
    realT * work1;
    realT * work2;
    realT * n_h;

    csrmv<I,T> * matvec;

    expm_multiply_gpu(const I,const int,const int,const realT,const T,const I,const I[],const I[],const T[]);
    ~expm_multiply_gpu() {
        if(B1)         cudaErrorCheck(cudaFree((void*)B1   ));
        if(B2)         cudaErrorCheck(cudaFree((void*)B2   ));
        if(F)          cudaErrorCheck(cudaFree((void*)F    ));
        if(work1)      cudaErrorCheck(cudaFree((void*)work1));
        if(work2)      cudaErrorCheck(cudaFree((void*)work2));
        if(n_h)        cudaErrorCheck(cudaFreeHost((void*)n_h));
        delete matvec;
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
    }

    void load_F(const T[]);
    void get_F(T[]);
};



template<typename I,typename realT,typename T>
expm_multiply_gpu<I,realT,T>::expm_multiply_gpu(const I _n_rows,const int _s,const int _m_star,
    const realT _tol,const T _eta,const I _nnz,const I Ap_host[],const I Aj_host[],const T Ax_host[]):
    n_rows(_n_rows), s(_s),m_star(_m_star), tol(_tol), eta(_eta)
{
    

    minBlocks  = (_n_rows + (__MAX_THREADS_INF_NORM * 2 - 1)) / (__MAX_THREADS_INF_NORM * 2);


    B1 = NULL;
    B2 = NULL;
    F  = NULL;
    work1 = NULL;
    work2 = NULL;
    n_h = NULL;
    // allocate gpu arrays
    cudaErrorCheck(cudaMalloc((void **)&B1,sizeof(T)*(_n_rows)));
    cudaErrorCheck(cudaMalloc((void **)&B2,sizeof(T)*(_n_rows)));
    cudaErrorCheck(cudaMalloc((void **)&F ,sizeof(T)*(_n_rows)));
    cudaErrorCheck(cudaMalloc((void **)&work1,sizeof(realT)*minBlocks));
    cudaErrorCheck(cudaMalloc((void **)&work2,sizeof(realT)*minBlocks));
    cudaErrorCheck(cudaMallocHost((void **)&n_h,sizeof(realT)*minBlocks));


    matvec = new csrmv_mp<I,T>(_n_rows,_n_rows,_nnz,Ap_host,Aj_host,Ax_host);
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

}



template<typename I,typename realT,typename T>
void expm_multiply_gpu<I,realT,T>::load_F(const T F_host[]){
    cudaErrorCheck(cudaMemcpy(F,F_host,sizeof(T)*(n_rows),cudaMemcpyHostToDevice));
}

template<typename I,typename realT,typename T>
void expm_multiply_gpu<I,realT,T>::get_F(T F_host[]){
    cudaErrorCheck(cudaMemcpy(F_host,F,sizeof(T)*(n_rows),cudaMemcpyDeviceToHost));
}

template<typename I,typename T>
__global__ void addcopy(const I N,T* F,T* B1,T* B2){
    const I k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k<N){
        F[k] += B1[k] = B2[k];
    }
}

template<typename I,typename T>
__global__ void equal(const I N,T* B1,T* B2){
    const I k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k<N){
        B1[k]=B2[k];
    }
}

template<typename I,typename T>
__global__ void addinplace(const I N,T* F,T* B2){
    const I k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k<N){
        F[k] += B2[k];
    }
}

template<typename T,typename I>
__global__ void muleq(const I N,T* F,T* B1,T a){
    const I k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k<N){
        F[k] *= a;
        B1[k] = F[k];
    }
}

template<typename T,typename I>
__global__ void mulinplace(const I N,T* F,T a){
    const I k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k<N){
        F[k] *= a;
    }
}


template<typename I,typename realT,typename T>
cudaError_t expm_multiply_core(expm_multiply_gpu<I,realT,T> * expm_obj,const int threads)
{

    // assign local vars
    const int s = expm_obj->s;
    const int m_star = expm_obj->m_star;
    const int blocks = (expm_obj->n_rows + (threads - 1)) / threads;
    const realT tol = expm_obj->tol;
    const T eta = expm_obj->eta;
    const I n_rows = expm_obj->n_rows;
    cudaStream_t s1 = expm_obj->s1;
    cudaStream_t s2 = expm_obj->s2;

    T* F = expm_obj->F;
    T* B1 = expm_obj->B1;
    T* B2 = expm_obj->B2;
    realT* work1 = expm_obj->work1;
    realT* work2 = expm_obj->work2;
    realT* n_h  = expm_obj->n_h;

    // create a CUDA stream 
    cudaEvent_t e0,e1;

    expm_obj->matvec->set_stream(s1);
    cudaEventCreateWithFlags(&e0, cudaEventDisableTiming);
    cudaEventRecord(e0,s2);

    for(int i=0;i<s;i++){
        // set B2=0 and find infinite norm at the same time
        // because inf_norm is using stream s1, this will not start 
        // until copy is finished from F to B1;

        cudaMemcpyAsync(B1,F,n_rows*sizeof(T),cudaMemcpyDeviceToDevice,s1);

        cudaStreamWaitEvent(s2,e0,0);
        inf_norm<I,realT,T>(n_rows,F,work1,n_h,s2);
    
		cudaEventDestroy(e0);
       	
        for(int j=1;j<m_star+1;j++){
            T step = 1.0/realT(j*s);
            cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
            // B2 -> A * B1
            (*expm_obj->matvec)(B1,B2,step);

            cudaMemcpyAsync(B1,B2,n_rows*sizeof(T),cudaMemcpyDeviceToDevice,s1);
          

            cudaEventRecord(e1,s1);
            cudaStreamWaitEvent(s2,e1,0);

            addinplace<<<blocks,threads,0,s2>>>(n_rows,F,B2);
            inf_norm<I,realT,T>(n_rows,B2,work1,n_h+1,s1); 
            inf_norm<I,realT,T>(n_rows,F ,work2,n_h+2,s2);

            cudaDeviceSynchronize();
            cudaEventDestroy(e1);
            
            if((n_h[0]+n_h[1])<=(tol*n_h[2])){
                break;
            }
            n_h[0] = n_h[1];
        }
    	cudaEventCreateWithFlags(&e0, cudaEventDisableTiming);
        mulinplace<<<blocks,threads,0,s1>>>(n_rows,F,eta);
        cudaEventRecord(e0,s1);
        
	}
	return cudaThreadSynchronize();
}



/*
template<typename I,typename realT,typename T>
cudaError_t expm_multiply_core(expm_multiply_gpu<I,realT,T> * expm_obj,const int threads)
{

    // assign local vars
    const int s = expm_obj->s;
    const int m_star = expm_obj->m_star;
    const int maxThreads = expm_obj->maxThreads;
    const int blocks = (expm_obj->n_rows + (threads - 1)) / threads;
    const realT tol = expm_obj->tol;
    const T eta = expm_obj->eta;
    const I n_rows = expm_obj->n_rows;

    T* F = expm_obj->F;
    T* B1 = expm_obj->B1;
    T* B2 = expm_obj->B2;
    realT* work1 = expm_obj->work1;
    realT* work2 = expm_obj->work2;
    realT* n_h  = NULL;

    cudaErrorCheck(cudaMallocHost((void **)&n_h,sizeof(realT)*3));

    // create a CUDA stream 
    cudaEvent_t e1;
    cudaStream_t s1,s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    int ncsrmv=0,nnorm=0;
    expm_obj->matvec->set_stream(s1);

    for(int i=0;i<s;i++){
        // set B2=0 and find infinite norm at the same time
        // because inf_norm is using stream s1, this will not start 
        // until copy is finished from F to B1;

        cudaMemsetAsync(B2,0,n_rows*sizeof(T),s1);
        cudaMemcpyAsync(B1,F,n_rows*sizeof(T),cudaMemcpyDeviceToDevice,s1);
        inf_norm<I,realT,T>(maxThreads,n_rows,F,work1,n_h,s2);
        nnorm++;
       
        for(int j=1;j<m_star+1;j++){
            T step = 1.0/realT(j*s);
            cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
            // B2 -> A * B1
            (*expm_obj->matvec)(B1,B2,step);
            ncsrmv++;

            cudaEventRecord(e1,s1);
            cudaStreamWaitEvent(s2,e1,0);

            addinplace<<<blocks,threads,0,s2>>>(n_rows,F,B2);
            cudaMemcpyAsync(B1,B2,n_rows*sizeof(T),cudaMemcpyDeviceToDevice,s1);
            inf_norm<I,realT,T>(maxThreads,n_rows,F, work1,n_h+2,s2);
            inf_norm<I,realT,T>(maxThreads,n_rows,B2,work2,n_h+1,s1); 
            nnorm+=2;

            cudaDeviceSynchronize();
            cudaEventDestroy(e1);

            if((n_h[0]+n_h[1])<=(tol*n_h[2])){
                break;
            }
        }
        mulinplace<<<blocks,threads,0,s2>>>(n_rows,F,eta);
        

    }
    cudaDeviceSynchronize();
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFreeHost(n_h);
    printf("%d  %d\n", ncsrmv,nnorm);
    return cudaThreadSynchronize();
}
*/

/*
template<typename I,typename realT,typename T>
cudaError_t expm_multiply_core(expm_multiply_gpu<I,realT,T> * expm_obj,const int threads)
{

    // assign local vars
    const int s = expm_obj->s;
    const int m_star = expm_obj->m_star;
    const int maxThreads = expm_obj->maxThreads;
    const int blocks = (expm_obj->n_rows + (threads - 1)) / threads;
    const realT tol = expm_obj->tol;
    const T eta = expm_obj->eta;
    const I n_rows = expm_obj->n_rows;

    T* F = expm_obj->F;
    T* B1 = expm_obj->B1;
    T* B2 = expm_obj->B2;
    realT* work1 = expm_obj->work1;
    realT* work2 = expm_obj->work2;
    realT* n_h  = NULL;

    cudaErrorCheck(cudaMallocHost((void **)&n_h,sizeof(realT)*3));

    // create a CUDA stream 
    cudaEvent_t e0,e1;
    cudaStream_t s1,s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    int ncsrmv = 0;

    expm_obj->matvec->set_stream(s1);

    for(int i=0;i<s;i++){
        // set B2=0 and find infinite norm at the same time
        // because inf_norm is using stream s1, this will not start 
        // until copy is finished from F to B1;
        cudaEventCreateWithFlags(&e0, cudaEventDisableTiming);

        cudaMemsetAsync(B2,0,n_rows*sizeof(T),s1);

        cudaMemcpyAsync(B1,F,n_rows*sizeof(T),cudaMemcpyDeviceToDevice,s2);
        cudaEventRecord(e0,s2);

        cudaStreamWaitEvent(s1,e0,0);       

       
        for(int j=1;j<m_star+1;j++){
            T step = 1.0/realT(j*s);
            cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
            ncsrmv++;
            // B2 -> A * B1
            (*expm_obj->matvec)(B1,B2,step);

            cudaEventRecord(e1,s1);
            cudaStreamWaitEvent(s2,e1,0);

            addinplace<<<blocks,threads,0,s2>>>(n_rows,F,B2);
            cudaMemcpyAsync(B1,B2,n_rows*sizeof(T),cudaMemcpyDeviceToDevice,s1);

            cudaEventDestroy(e1);

        }
        mulinplace<<<blocks,threads,0,s2>>>(n_rows,F,eta);

        cudaEventDestroy(e0);
    }
    cudaDeviceSynchronize();
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFreeHost(n_h);
    printf("%d\n", ncsrmv);
    return cudaThreadSynchronize();
}
*/


#endif