import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy import optimize

def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)


# mimic your data
#x = np.linspace(0, 5721)
#y = 50 - np.clip(x, 10, 40)

x = np.array(range(0,5721))
y = random.rand(5721)

print(x.shape)
print(y.shape)

# apply the segment fit
fx, fy = segments_fit(x, y, 3)

print(fx.shape)
print(fy.shape)

# show the results
plt.figure(figsize=(8, 3))
plt.plot(fx, fy, 'o-')
plt.plot(x, y, '.')
plt.legend(['fitted line', 'given points'])
plt.show()
