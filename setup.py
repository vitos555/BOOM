import os
import subprocess
from setuptools import setup
from setuptools import Extension

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

INCLUDE_DIRS = ['.', './Bmath', './math/cephes']

try:
    import numpy
    HAS_NUMPY = True
    INCLUDE_DIRS += [numpy.get_include()]
except ImportError:
    HAS_NUMPY = False

file_ext = '.pyx' if HAS_CYTHON else '.cpp'

def find_cpp(path, recursive=True, exclude=[]):
    ret = []
    if recursive:
        for root, subdirs, files in os.walk(path):
            for exclusion in exclude:
                if exclusion in root:
                    continue
            for file in files:
                if file[-4:] == ".cpp":
                    ret.append(os.path.join(root, file))
    else:
        for file in os.listdir(path):
            for exclusion in exclude:
                if exclusion in file:
                    continue
            if file[-4:] == ".cpp":
                ret.append(os.path.join(path, file))

    return ret

if not os.path.exists("boost_1_68_0"):
    subprocess.run(["curl", "-L", "-O", "https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz"])
    subprocess.run(["tar", "-xf", "boost_1_68_0.tar.gz"])

INCLUDE_DIRS += ["./boost_1_68_0"]

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
     find_cpp("Models/") + \
     find_cpp(".", recursive=False) + \
     find_cpp("LinAlg/") + \
     find_cpp("math/cephes/") + \
     find_cpp("numopt/") + \
     find_cpp("Bmath/") + \
     find_cpp("Samplers/") + \
     find_cpp("stats/") + \
     find_cpp("TargetFun/") + \
     find_cpp("cpputil/"),
    include_dirs=INCLUDE_DIRS,
    language="c++",
    libraries=[],
    library_dirs=[],
    extra_compile_args=['-std=c++11', '-DADD_'] + ["-I"+include_dir for include_dir in INCLUDE_DIRS],
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
    package_dir={'causal_impact': 'Interfaces/Python/causal_impact'},
)
