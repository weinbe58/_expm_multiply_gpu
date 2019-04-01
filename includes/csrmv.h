#ifndef __CSR_H__
#define __CSR_H__

#include <cuda_runtime.h>
#include <cusparse.h>
#include "error.h"

#ifndef __complex_types__
#define __complex_types__

#include <thrust/complex.h>
#include <cuComplex.h>

typedef thrust::complex<double> cdouble;
typedef thrust::complex<T> cfloat;

#endif

template<typename I,typename T>
class csrmv
{
public:
	cusparseHandle_t cusparseH;
  cusparseMatDescr_t descrA;
  cudaStream_t stream;
	T* Ax;
	I* Ap;
	I* Aj;
	const I n_rows;
	const I n_cols;
	const I nnz;


	csrmv(const I nr,const I nc, const I nz,const I Ap_host[],const I Aj_host[],const T Ax_host[]) :
	n_rows(nr), n_cols(nc), nnz(nz)
	{
		stream    = (cudaStream_t)0;
		descrA    = NULL;
		cusparseH = NULL;
		Ap        = NULL;
		Aj        = NULL;
		Ax        = NULL;
		cudaErrorCheck(cudaMalloc((void **)&Ap,sizeof(I)*(nr+1)));
		cudaErrorCheck(cudaMalloc((void **)&Aj,sizeof(I)*(nz  )));
		cudaErrorCheck(cudaMalloc((void **)&Ax,sizeof(T)*(nz  )));

		cudaErrorCheck(cudaMemcpy(Ap,Ap_host,sizeof(I)*(nr+1),cudaMemcpyHostToDevice));
		cudaErrorCheck(cudaMemcpy(Aj,Aj_host,sizeof(I)*(nz  ),cudaMemcpyHostToDevice));
		cudaErrorCheck(cudaMemcpy(Ax,Ax_host,sizeof(T)*(nz  ),cudaMemcpyHostToDevice));


    cusparseErrorCheck(cusparseCreate(&cusparseH));
    cusparseErrorCheck(cusparseCreateMatDescr(&descrA));

    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

	}

	~csrmv(){
		if(Ap        ) cudaErrorCheck(cudaFree((void*)Ap));
		if(Aj        ) cudaErrorCheck(cudaFree((void*)Aj));
		if(Ax        ) cudaErrorCheck(cudaFree((void*)Ax));
		if (descrA   ) cusparseDestroyMatDescr(descrA);
		if (cusparseH) cusparseDestroy(cusparseH);

	}

	void set_stream(cudaStream_t stream){
	    cusparseErrorCheck(cusparseSetStream(cusparseH, stream));
	}

	virtual void operator()(T*,T*,T) = 0;

};


template<typename T>
inline cudaDataType get_typecode(void);

template<>
inline cudaDataType get_typecode<float>(void){
  return CUDA_R_32F;
}

template<>
inline cudaDataType get_typecode<cfloat>(void){
  return CUDA_C_32F;
}

template<>
inline cudaDataType get_typecode<double>(void){
  return CUDA_R_64F;
}

template<>
inline cudaDataType get_typecode<cdouble>(void){
  return CUDA_C_64F;
}

template<typename I,typename T>
class csrmv_mp : public csrmv<I,T>
{
  const T h_zero;
  size_t bufferSizeInBytes;
  void * buffer;
  cudaDataType type;
public:

    csrmv_mp(const I nr,const I nc, const I nz,const I Ap_host[],const I Aj_host[],const T Ax_host[]) : 
    csrmv<I,T>::csrmv(nr,nc,nz,Ap_host,Aj_host,Ax_host), h_zero(0.0), type(get_typecode<T>())  { 
        T h_one=1.0;
        
        cusparseErrorCheck(
            cusparseCsrmvEx_bufferSize(csrmv<I,T>::cusparseH, 
                                       CUSPARSE_ALG_MERGE_PATH,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                       csrmv<I,T>::n_rows, 
                                       csrmv<I,T>::n_cols, 
                                       csrmv<I,T>::nnz,
                                       (const void *)&h_one, type,
                                       csrmv<I,T>::descrA,
                                       (const void *)csrmv<I,T>::Ax, type,
                                       csrmv<I,T>::Ap,
                                       csrmv<I,T>::Aj,
                                       NULL, type,
                                       (const void *)&h_zero,type,
                                       NULL, type,
                                       type,
                                       &bufferSizeInBytes));
        cudaErrorCheck(cudaMalloc(&buffer,bufferSizeInBytes));

    }
    ~csrmv_mp(){
        if(buffer)  cudaErrorCheck(cudaFree(buffer));
    }
    void operator()(T* x,T* y,T a){
        cusparseErrorCheck(
          cusparseCsrmvEx(csrmv<I,T>::cusparseH, 
                         CUSPARSE_ALG_MERGE_PATH,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, 
                         csrmv<I,T>::n_rows, 
                         csrmv<I,T>::n_cols, 
                         csrmv<I,T>::nnz,
                         (const void *)&a, type,
                         csrmv<I,T>::descrA,
                         (const void *)csrmv<I,T>::Ax, type,
                         csrmv<I,T>::Ap,
                         csrmv<I,T>::Aj,
                         (const void *)x, type,
                         (const void *)&h_zero,type,
                         (void *)y, type,
                         type, buffer));
    }

};



#endif