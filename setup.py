import os
import sys
import subprocess
import glob
from sysconfig import get_paths

def build_static_lib():
    package_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = get_paths()["data"]

    package_dir = os.path.expandvars(package_dir)
    data_path = os.path.expandvars(data_path)
    script = os.path.join(package_dir,'build_lib_dynamic.sh')
    data_path = get_paths()["data"]
    nvcc_path = os.path.join(os.sep,"usr","bin","nvcc")
    cmd = ["bash",script,nvcc_path,package_dir]
    subprocess.check_call(cmd)



def cython_files():
    import os,glob
    from Cython.Build import cythonize

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    cython_src = glob.glob(os.path.join(package_dir,"*.pyx"))

    cythonize(cython_src)


def configuration(parent_package='', top_path=None):
    import numpy,os,sys,glob
    from numpy.distutils.misc_util import Configuration
    config = Configuration('_expm_multiply_gpu',parent_package, top_path)

    cython_files()
    build_static_lib()
    
    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    src = os.path.join(package_dir,"expm_multiply_cuda_py.cpp")
    library_dirs = [os.path.join(package_dir,"lib")]

    extra_link_args = ["-lcudart","-lcusparse","-lexpm_multiply_cuda"]
    config.add_extension('expm_multiply_cuda_py',sources=src,
                            extra_link_args=extra_link_args,
                            library_dirs=library_dirs,
                            runtime_library_dirs=library_dirs,
                            language="c++")

    return config

if __name__ == '__main__':
        from numpy.distutils.core import setup
        setup(**configuration(top_path='').todict())



