import causal_impact
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0, 4.5, 3.5, 2.5, 2.6])
X = np.array([[1.0, 2.0, 0.0, 0.0, 0.0, 3.5, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0, 4.4, 0.0, 2.5, 2.5]])
                 
b = causal_impact.CausalImpact(X, y, range(0, 5), range(6, 8), niter=1000, burn=100, seed=1, 
							   seasons=[])
res = b.analyze()
print(res[0], res[1])
print(b.summary())
#print(b.bsts.crepr())
