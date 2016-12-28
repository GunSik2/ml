"""Softmax."""
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    expx = np.exp(x)
    sum = np.sum(expx, axis=0)
    return np.divide(expx, sum)


scores = np.array([3.0, 1.0, 0.2])

print(softmax(scores))
print(softmax(scores * 10))
print(softmax(scores / 10)) 



# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
