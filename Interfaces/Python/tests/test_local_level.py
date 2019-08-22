import pybsts
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0, 4.5])

specification = {"ar_order": 1, "local_trend": {"local_level": True},
                 "sigma_prior": np.std(y, ddof=1), "initial_value": y[0]}
b = pybsts.PyBsts("gaussian", specification, {"ping": 10, "niter":100, "seed": 1})
b.fit(y, seed=1)
res = b.predict(seed=1)
print(res)
