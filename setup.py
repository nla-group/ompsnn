from setuptools import setup, Extension
import pybind11

snn_module = Extension(
    'snnomp',
    sources=['snnpy.cpp'],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp', '-lblas'],
    language='c++',
)

setup(
    name='snnomp',
    version='1.0',
    description='SNN library with OpenMP optimization',
    ext_modules=[snn_module],
)