# cython: language_level=2
# distutils: language=c++

from numpy cimport ndarray,PyArray_DATA,npy_intp
from libcpp cimport bool

from scipy.sparse import isspmatrix_csr,csr_matrix
from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo,_fragment_3_1
from scipy.sparse.construct import eye
import numpy as _np




cdef extern from "expm_multiply_cuda.h":
    void* expm_multiply_cuda_create(int,npy_intp,npy_intp,
            npy_intp,void*,void*,npy_intp,void*,void*,void*);
    void expm_multiply_cuda_destroy(void*,int);
    void expm_multiply_cuda_load_F(void*,int,void*);
    void expm_multiply_cuda_get_F(void*,int,void*);
    int  expm_multiply_cuda_core(void*,int,int);

cdef class expm_multiply_gpu:
    cdef void* expm_obj
    cdef int call_code
    cdef npy_intp n_rows
    cdef object dtype
    cdef bool vector_loaded


    def __cinit__(self,object A):
        if not isspmatrix_csr(A):
            A = csr_matrix(A)

        if A.shape[0] != A.shape[1]:
            raise ValueError("expecing square matrix.")

        A.asfptype()

        realdtype = _np.finfo(A.dtype).eps.dtype

        # calculating s,m_star
        tol = _np.finfo(A.dtype).eps/2
        mu = A.diagonal().sum() / float(A.shape[0])

        shift = eye(A.shape[0], A.shape[1],dtype=A.dtype, format="csr")
        shift.data *= mu
        
        A = A - shift
        A_1_norm = _np.max(_np.asarray(abs(A).sum(axis=0)).ravel())
        if A_1_norm == 0:
            m_star, s = 0, 1
        else:
            ell = 2
            norm_info = LazyOperatorNormInfo(A, A_1_norm=A_1_norm, ell=ell)
            m_star, s = _fragment_3_1(norm_info, 1, tol, ell=ell)

        eta = _np.exp(mu/float(s))

        cdef ndarray tol_arr = _np.array(tol,dtype=realdtype)
        cdef ndarray eta_arr = _np.array(eta,dtype=A.dtype)
        cdef ndarray Ax = A.data
        cdef ndarray Aj = A.indices
        cdef ndarray Ap = A.indptr
        cdef npy_intp n_rows = A.shape[0]
        cdef npy_intp nnz = A.data.shape[0]

        if A.dtype == _np.complex128:
            self.call_code = 0
        elif A.dtype == _np.float64:
            self.call_code = 1
        elif A.dtype == _np.complex64:
            self.call_code = 2
        elif A.dtype == _np.float32:
            self.call_code = 3
        else:
            raise TypeError("A must contain floating point data.")

        self.vector_loaded=False
        self.n_rows = n_rows
        self.dtype = A.dtype
        self.expm_obj = expm_multiply_cuda_create(self.call_code,n_rows,<int>s,<int>m_star,
            PyArray_DATA(tol_arr),PyArray_DATA(eta_arr),nnz,PyArray_DATA(Ap),PyArray_DATA(Aj),PyArray_DATA(Ax))
        
    def __dealloc__(self):
        # cdef void* expm_obj = self.expm_obj
        expm_multiply_cuda_destroy(self.expm_obj,self.call_code)


    def load_vector(self,object y):
        cdef ndarray y_arr = _np.asarray(y,dtype=self.dtype)
        self.vector_loaded=True
        if y_arr.ndim != 1:
            raise ValueError("input array must be 1-D array.")

        if y_arr.shape[0] != self.n_rows:
            errstr = "input array shape ({},) does not match matrix shape ({},{})"
            raise ValueError(errstr.format(y_arr.shape[0],self.n_rows,self.n_rows))

        expm_multiply_cuda_load_F(self.expm_obj,self.call_code,PyArray_DATA(y_arr))

    def get_result(self,object out=None):
        cdef ndarray out_arr

        if out is not None:
            if not isinstance(out,_np.ndarray):
                raise TypeError("out must be a numpy array")

            if out.size != self.n_rows:
                errstr = "out array must have at least {} elements to store result"
                raise ValueError(errstr.format(self.n_rows))

            if not _np.can_cast(out.dtype,self.dtype,casting="no"):
                errstr = "out array dtype {} does not match result dtype {}"
                raise ValueError(errstr.format(out.dtype,self.dtype))

            if not out.flags["CARRAY"]:
                raise ValueError("output array must be C-contiguous and writable.")

            out_arr = out
        else:
            out_arr = _np.zeros(self.n_rows,dtype=self.dtype)


        expm_multiply_cuda_get_F(self.expm_obj,self.call_code,PyArray_DATA(out_arr))

        return out_arr


    def run_expm_multiply(self,int nthreads=1024):
        if not self.vector_loaded:
            raise RuntimeError("exponential being calculated when \
                no vector has been loaded loaded to gpu.")
        if nthreads not in [1,2,4,8,16,32,64,128,256,512,1024]:
            raise ValueError("nthreads must be a power of 2 from 1 up to 512.")

        return expm_multiply_cuda_core(self.expm_obj,self.call_code,nthreads)
