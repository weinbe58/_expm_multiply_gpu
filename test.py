from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
from quspin.tools.evolution import expm_multiply_parallel
from expm_multiply_cuda_py import expm_multiply_gpu

from scipy.sparse.linalg import expm_multiply
from scipy.sparse import random,identity
import numpy as np
from cProfile import Profile

# np.random.seed(0)

L=22

J = [[1.0,i,(i+1)%L] for i in range(L)]
static = [["xx",J],["yy",J],["zz",J]]
basis = spin_basis_general(L,m=0)
print(basis.Ns)
H = hamiltonian(static,[],basis=basis,dtype=np.float64)



a = -10j

A = a*H.static

niter = 1



v = np.random.normal(0,1,size=basis.Ns)+1j*np.random.normal(0,1,size=basis.Ns)
v /= np.linalg.norm(v)

work = np.zeros(2*basis.Ns,dtype=np.complex128)

U1 = expm_multiply_gpu(A)
U2 = expm_multiply_parallel(H.static,a=a)

pr=Profile()
pr.enable()
U1.load_vector(v)
for i in range(niter):
	U1.run_expm_multiply(1024)
r1 = U1.get_result()
pr.disable()
pr.print_stats(sort="time")

r2 = v.copy()
pr=Profile()
pr.enable()
for i in range(niter):
	U2.dot(r2,work_array=work,overwrite_v=True)
pr.disable()
pr.print_stats(sort="time")

r3 = v.copy()
pr=Profile()
pr.enable()
for i in range(niter):
	r3 = expm_multiply(A,r3)
pr.disable()
pr.print_stats(sort="time")


print(np.linalg.norm(r2-r1))
print(np.linalg.norm(r3-r1))
