import pybsts
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0, 4.5])
X = np.array([[1.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0, 0.0]])


specification = {"local_trend": {"static_intercept": True},
                 "predictors_prior": {"predictors_squared_normalized": np.dot(X, X.T)/X.shape[1]},
                 "sigma_prior": np.std(y, ddof=1), "initial_value": y[0], "mean_value": np.mean(y), 
                 "predictor_names": ["first", "second"]}
b = pybsts.PyBsts("gaussian", specification, 
                  {"ping": 10, "niter":100, "burn": 10, "forecast_horizon": 2})
print(b.crepr().decode("UTF-8"))
b.fit(X, y, seed=1)
#print(b.crepr().decode("UTF-8"))
res = b.predict(np.array([[1.0, 0.0], [2.0, 0.0]]), [6, 7], seed=1)
print(res)
#print(b.crepr().decode("UTF-8"))
print(b.results("state.contributions"))
