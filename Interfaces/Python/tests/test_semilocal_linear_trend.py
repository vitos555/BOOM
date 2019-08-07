import pybsts
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0, 4.5])

specification = {"ar_order": 1, "local_trend": {"semilocal_linear_trend": True},
                 "sigma_prior": np.std(y, ddof=1), "initial_value": y[0]}
b = pybsts.PyBsts("gaussian", specification, {"ping": 10, "niter":100, "seed": 1})
#print(b.crepr().decode("UTF-8"))
b.fit(y)
#print(b.crepr().decode("UTF-8"))
res = b.predict()
print(res)
#print(b.crepr().decode("UTF-8"))
