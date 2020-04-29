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
            found_exclusion = False
            for exclusion in exclude:
                if exclusion in root:
                    found_exclusion = True
            if found_exclusion:
                continue
            for file in files:
                if file[-4:] == ".cpp":
                    ret.append(os.path.join(root, file))
    else:
        for file in os.listdir(path):
            found_exclusion = False
            for exclusion in exclude:
                if exclusion in file:
                    found_exclusion = True
            if found_exclusion:
                continue
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
     find_cpp("Models/", exclude=["Bart", "Nnet", "test", "HMM", "IRT", "Mixtures"]) + \
     find_cpp(".", recursive=False) + \
     find_cpp("LinAlg/") + \
     find_cpp("math/") + \
     find_cpp("numopt/") + \
     find_cpp("Bmath/") + \
     find_cpp("Samplers/", exclude=["failed_experiments"]) + \
     find_cpp("stats/", exclude=["test"]) + \
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

long_description="""
About PyBSTS
============

PyBSTS is an adaptation of R's implementation of Steven L. Scott's 
[BSTS library](https://cran.r-project.org/web/packages/bsts/). 
It has similar interface, but re-written for Python memory model. 
It is a Cython+Numpy based implementation and thus dependendecies for these packages.

Package Contents
----------------

PyBSTS package installs pybsts and causal_impact libraries. 

Quick start
-----------

1. Install the package
```
pip install pybsts
```

2. Build BSTS model
```
import pybsts
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0, 4.5])

specification = {"ar_order": 1, "local_trend": {"local_level": True},
                 "sigma_prior": np.std(y, ddof=1), "initial_value": y[0]}
b = pybsts.PyBsts("gaussian", specification, {"ping": 10, "niter":100, "seed": 1, "burn": 10})
b.fit(y, seed=1)
res = b.predict(seed=1)
print(res)

y = np.array([1.0, 2.0, 3.0, 4.0, 4.5])
X = np.array([[1.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0, 0.0]])


specification = {"local_trend": {"static_intercept": True},
                 "predictors_prior": {"predictors_squared_normalized": np.dot(X, X.T)/X.shape[1]},
                 "sigma_prior": np.std(y, ddof=1), "initial_value": y[0], "mean_value": np.mean(y), 
                 "predictor_names": ["first", "second"]}
b = pybsts.PyBsts("gaussian", specification, 
                  {"ping": 10, "niter":100, "burn": 10, "forecast_horizon": 2, "seed": 1})
b.fit(X.T, y, seed=1)
res = b.predict(np.array([[1.0, 0.0], [2.0, 0.0]]), [6, 7], seed=1)
print(res)
```

3. Build CausalImpact model
```
import causal_impact
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0, 4.5, 3.5, 2.5, 2.6])
X = np.array([[1.0, 2.0, 0.0, 0.0, 0.0, 3.5, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0, 4.4, 0.0, 2.5, 2.5]])
                 
b = causal_impact.CausalImpact(X, y, range(0, 5), range(6, 8), niter=1000, burn=100, seed=1, 
                               seasons=[{"number_of_seasons": 3, "duration": 1}])
res = b.analyze()
print(res[0], res[1])
print(b.summary())
```

Current status
--------------

Here is a list of implemented models (see [BSTS Library Documentation](https://cran.r-project.org/web/packages/bsts/bsts.pdf) ):
- stationary (non-dynamic) gaussian regression with local.level + seasons
- stationary (non-dynamic) gaussian regression with local.linear.trend + seasons
- stationary (non-dynamic) gaussian regression with semilocal.linear.trend + seasons
- stationary (non-dynamic) gaussian regression with static.intercept + seasons
- any of the above + ar
- any of the above + auto.ar
"""

setup(
    name="pybsts",
    author="Vitalii Ostrovskyi",
    author_email="vitos@vitos.org.ua",
    description="Python interface to Bayesian Structured Time Series",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vitos555/BOOM",
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
    ],
    license="LGPL 2.1",
    keywords="bsts,pybsts,bayesian time series,time series,prediction",
    version='1.0.7',
    include_package_data=True,
    ext_modules=extensions,
    install_requires=['cython', 'numpy'],
    packages=['causal_impact'],
    package_dir={'causal_impact': 'Interfaces/Python/causal_impact'},
    python_requires='>=3.6',
)
