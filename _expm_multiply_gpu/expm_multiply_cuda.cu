



#include <stdint.h>
#include "expm_multiply_cuda_src.h"


typedef int64_t npy_intp;
typedef int32_t npy_int32;

typedef expm_multiply_gpu<npy_int32,double,cdouble> expm_cdouble;
typedef expm_multiply_gpu<npy_int32,double,double> expm_double;
typedef expm_multiply_gpu<npy_int32,float,cfloat> expm_cfloat;
typedef expm_multiply_gpu<npy_int32,float,float> expm_float;


extern "C" {



void* expm_multiply_cuda_create(int call_code,
								npy_intp n_rows,
								int s,
								int m_star,
								void* tol,
								void* eta,
								npy_intp nnz,
								void* Ap_host,
								void* Aj_host,
								void* Ax_host)
{
	void* expm_obj=NULL;
	switch(call_code){
		case 0:
			expm_obj=(void*)new expm_cdouble((npy_int32)n_rows,s,m_star, \
				*((double*)tol),*((cdouble*)eta),(npy_int32)nnz,(npy_int32*)Ap_host,(npy_int32*)Aj_host,(cdouble*)Ax_host);
			break;
		case 1:
			expm_obj=(void*)new expm_double((npy_int32)n_rows,(int)s,(int)m_star, \
				*((double*)tol),*((double*)eta),(npy_int32)nnz,(npy_int32*)Ap_host,(npy_int32*)Aj_host,(double*)Ax_host);
			break;
		case 2:
			expm_obj=(void*)new expm_cfloat((npy_int32)n_rows,(int)s,(int)m_star, \
				*((float*)tol),*((cfloat*)eta),(npy_int32)nnz,(npy_int32*)Ap_host,(npy_int32*)Aj_host,(cfloat*)Ax_host);
			break;
		case 3:
			expm_obj=(void*)new expm_float((npy_int32)n_rows,(int)s,(int)m_star, \
				*((float*)tol),*((float*)eta),(npy_int32)nnz,(npy_int32*)Ap_host,(npy_int32*)Aj_host,(float*)Ax_host);
			break;
		default:
			break;
	}
	return expm_obj;
}


void expm_multiply_cuda_destroy(void* expm_obj,int call_code)
{
	switch(call_code){
		case 0:
			delete ((expm_cdouble*)expm_obj);
			break;
		case 1:
			delete ((expm_double*)expm_obj);
			break;
		case 2:
			delete ((expm_cfloat*)expm_obj);
			break;
		case 3:
			delete ((expm_float*)expm_obj);
			break;
	}
}


void expm_multiply_cuda_load_F(void* expm_obj,int call_code,void* F_host)
{
	switch(call_code){
		case 0:
			((expm_cdouble*)expm_obj)->load_F((cdouble*)F_host);
			break;
		case 1:
			((expm_double*)expm_obj)->load_F((double*)F_host);
			break;
		case 2:
			((expm_cfloat*)expm_obj)->load_F((cfloat*)F_host);
			break;
		case 3:
			((expm_float*)expm_obj)->load_F((float*)F_host);
			break;
	}
}

void expm_multiply_cuda_get_F(void* expm_obj,int call_code,void* F_host)
{
	switch(call_code){
		case 0:
			((expm_cdouble*)expm_obj)->get_F((cdouble*)F_host);
			break;
		case 1:
			((expm_double*)expm_obj)->get_F((double*)F_host);
			break;
		case 2:
			((expm_cfloat*)expm_obj)->get_F((cfloat*)F_host);
			break;
		case 3:
			((expm_float*)expm_obj)->get_F((float*)F_host);
			break;
	}
}


int expm_multiply_cuda_core(void* expm_obj,int call_code,int threads)
{
	switch(call_code){
		case 0:
			return (int) expm_multiply_core<npy_int32,double,cdouble>((expm_cdouble*)expm_obj,threads);
		case 1:
			return (int) expm_multiply_core<npy_int32,double,double>((expm_double*)expm_obj,threads);
		case 2:
			return (int) expm_multiply_core<npy_int32,float,cfloat>((expm_cfloat*)expm_obj,threads);
		case 3:
			return (int) expm_multiply_core<npy_int32,float,float>((expm_float*)expm_obj,threads);
	}
	return -1;
}

}