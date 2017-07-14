import numpy as np

a = np.array(
    [
        [1., 0., 0.],
        [0., 1., 2.]
    ],
    dtype=float
)
print type(a)
print a.ndim, a.shape, a.size, a.dtype, a.itemsize

# create sequences of numbers
print np.arange(10, 30, 5)
print np.linspace(0, 2, 9)

print np.zeros((2, 3, 4, 5))
print np.ones((2, 3, 4), dtype=np.int16)  # dtype can also be specified
print np.empty((2, 3))  # uninitialized, output may vary

print "============ Basic Operations ================"
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print a
print 10 * np.sin(a)
print a < 35

print b
print b ** 2

print a - b

