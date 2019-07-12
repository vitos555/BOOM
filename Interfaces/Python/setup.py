import os
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

file_ext = '.pyx' if HAS_CYTHON else '.cpp'

extensions = [Extension("pybsts", 
    ["bsts/pybsts" + file_ext,
     "bsts/model_manager.cpp",
     "bsts/state_space_gaussian_model_manager.cpp",
     "bsts/state_space_regression_model_manager.cpp",
     "bsts/prior_specification.cpp",
     "bsts/spike_slab_prior.cpp"],
    include_dirs=['../../'],
    language="c++",
    libraries=['c++'],
    library_dirs=[],
    extra_objects=['../../libboom.a'],
    extra_compile_args=['-std=c++11'],
    extra_link_args=['-mmacosx-version-min=10.13'])]

if HAS_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, compiler_directives={'language_level': 3})

setup(
    name="pybsts",
    author="Vitalii Ostrovskyi",
    author_email="vitos@vitos.org.ua",
    description="Python interface to Bayesian Structured Time Series",
    version='0.0.1',
    ext_modules=extensions,
)
