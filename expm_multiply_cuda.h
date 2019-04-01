#ifndef __EXPM_MULTIPLY_CUDA__
#define __EXPM_MULTIPLY_CUDA__

#include <stdint.h>
typedef int64_t npy_intp;
typedef int npy_int32;


extern "C" {


void* expm_multiply_cuda_create(int,npy_intp,npy_intp,
	npy_intp,void*,void*,npy_intp,void*,void*,void*);
void expm_multiply_cuda_destroy(void*,int);
void expm_multiply_cuda_load_F(void*,int,void*);
void expm_multiply_cuda_get_F(void*,int,void*);
int  expm_multiply_cuda_core(void*,int,int);


}


#endif