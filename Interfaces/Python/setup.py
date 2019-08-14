import os
from setuptools import setup
from setuptools import Extension

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

try:
    import numpy
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

file_ext = '.pyx' if HAS_CYTHON else '.cpp'

extensions = [Extension("pybsts", 
    ["bsts/pybsts" + file_ext,
     "bsts/model_manager.cpp",
     "bsts/state_space_gaussian_model_manager.cpp",
     "bsts/state_space_regression_model_manager.cpp",
     "bsts/prior_specification.cpp",
     "bsts/spike_slab_prior.cpp",
     "bsts/create_state_model.cpp",
     "bsts/list_io.cpp"],
    include_dirs=['../../'] + [numpy.get_include()] if HAS_NUMPY else [],
    language="c++",
    libraries=[],
    library_dirs=[],
    extra_objects=['../../libboom.a'],
    extra_compile_args=['-std=c++11', '-DADD_'],
    extra_link_args=[])]

if HAS_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, compiler_directives={'language_level': 3})

setup(
    name="pybsts",
    author="Vitalii Ostrovskyi",
    author_email="vitos@vitos.org.ua",
    description="Python interface to Bayesian Structured Time Series",
    version='1.0.2',
    ext_modules=extensions,
    install_requires=['cython', 'numpy'],
    packages=['causal_impact']
)
