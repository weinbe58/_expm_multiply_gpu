
#include <cuda_runtime.h>
#include "error.h"
#include "inf_norm_cuda.h"
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <ctime>



template<class T,class realT>
realT cpu_inf_norm(const int size,const T arr[]){
	realT inf_norm = 0.0;
	for(int i=0;i<size;++i){
		inf_norm = std::max(inf_norm,std::abs(arr[i]));
	}
	return inf_norm;
}




template<class T,class realT>
void test_inf_norm(int size,int niter){
    unsigned int threads = (size < 2*__MAX_THREADS_INF_NORM) ? nextPow2((size + 1)/ 2) : __MAX_THREADS_INF_NORM;
    unsigned int blocks  = (size + (threads * 2 - 1)) / (threads * 2);

	std::srand(std::time(nullptr));
	T * hi_data = NULL;
	T * di_data = NULL;
	realT * do_data = NULL;
	realT result_gpu,result_cpu;

	cudaErrorCheck(cudaMallocHost((void**)&hi_data,size*sizeof(T)));
	cudaErrorCheck(cudaMalloc((void**)&di_data,size*sizeof(T)));
	cudaErrorCheck(cudaMalloc((void**)&do_data,blocks*sizeof(realT)));
	for(int j=0;j<niter;++j)
	{	
		for(int i=0;i<size;++i){
				hi_data[i] = std::rand();
		}

		cudaErrorCheck(cudaMemcpy(di_data,hi_data,size*sizeof(T),cudaMemcpyHostToDevice));

		inf_norm<int,realT,T>(size,di_data,do_data,&result_gpu);
	    cudaDeviceSynchronize();

		result_cpu = cpu_inf_norm<T,realT>(size,hi_data);

		assert(result_cpu==result_gpu);
	}
	if(hi_data) cudaErrorCheck(cudaFreeHost(hi_data));
	if(di_data) cudaErrorCheck(cudaFree(di_data));
	if(do_data) cudaErrorCheck(cudaFree(do_data));



}





int main(int argc, char const *argv[])
{

	test_inf_norm<double,double>(70,1000);

	cudaErrorCheck(cudaDeviceReset());

	return 0;
}