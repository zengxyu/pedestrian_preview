import numpy as np

a = np.array([1, 2, 3, np.nan])
print(a)
a[np.isnan(a)] = 10
print(a)
