import numpy as np

scores = np.array([1., 2., 3.])
scores2 = np.vstack([scores, scores])

print(scores)
print(scores2)
print(np.ones_like(scores))
print(np.sum(scores))
print(np.sum(scores2, axis=0))
