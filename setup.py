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

def find_cpp(path):
    ret = []
    for file in os.listdir(path):
        if file[-4:] == ".cpp":
            ret.append(os.path.join(path, file))
    return ret


extensions = [Extension("pybsts", 
    ["Interfaces/Python/bsts/pybsts" + file_ext,
     "Interfaces/Python/bsts/model_manager.cpp",
     "Interfaces/Python/bsts/state_space_gaussian_model_manager.cpp",
     "Interfaces/Python/bsts/state_space_regression_model_manager.cpp",
     "Interfaces/Python/bsts/prior_specification.cpp",
     "Interfaces/Python/bsts/spike_slab_prior.cpp",
     "Interfaces/Python/bsts/create_state_model.cpp",
     "Interfaces/Python/bsts/list_io.cpp"] + \
     find_cpp("distributions/") + \
     find_cpp("Models/Glm/") + \
     find_cpp("Models/Glm/PosteriorSamplers/") + \
     find_cpp("Models/Hierarchical/") + \
     find_cpp("Models/HMM/") + \
     find_cpp(".") + \
     find_cpp("Models/IRT/") + \
     find_cpp("LinAlg/") + \
     find_cpp("math/cephes/") + \
     find_cpp("Models/Mixtures/") + \
     find_cpp("Models/") + \
     find_cpp("Models/Policies/") + \
     find_cpp("numopt/") + \
     find_cpp("Models/PointProcess/") + \
     find_cpp("Bmath/") + \
     find_cpp("Samplers/") + \
     find_cpp("Samplers/Gilks/") + \
     find_cpp("Models/StateSpace/") + \
     find_cpp("stats/") + \
     find_cpp("TargetFun/") + \
     find_cpp("Models/TimeSeries/") + \
     find_cpp("cpputil/"),
    include_dirs=['.', './Bmath', './math/cephes'] + numpy.get_include() if HAS_NUMPY else [],
    language="c++",
    libraries=[],
    library_dirs=[],
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
    version='1.0.0',
    ext_modules=extensions,
    install_requires=['cython', 'numpy'],
    packages=['causal_impact'],
    package_dir={'causal_impact': 'Interfaces/Python/'},
)
