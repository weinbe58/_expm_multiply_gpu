#ifndef __ERROR_H__
#define __ERROR_H__

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>


#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// this function copied from: http://berenger.eu/blog/cusparse-cccuda-sparse-matrix-examples-csr-bcsr-spmv-and-conversions/
static const char * cusparseGetErrorString(cusparseStatus_t error)
{   
    // Read more at: http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar
    switch (error)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return "The operation completed successfully.";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\n" \
               "To correct: call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
 
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\n"\
                "To correct: prior to the function call, deallocate previously allocated memory as much as possible.";
 
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n"\
            "To correct: ensure that all the parameters being passed have valid values.";
 
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\n"\
            "To correct: compile and run the application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic operations and 1.3 for double precision.";
 
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n"\
            "To correct: prior to the function call, unbind any previously bound textures.";
 
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n"\
                "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
 
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n"\
                "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";
 
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n"\
                "To correct: check that the fields in cusparseMatDescr_t descrA were set correctly.";
    }
 
    return "<unknown>";
}

#define cusparseErrorCheck(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
      fprintf(stderr,"CUSPARSE ERROR: %s %s %d\n", cusparseGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




#endif