import numpy as np

a = np.array([1, 2, 3, 4, 5])

a[a > 3] = 0

print(a)