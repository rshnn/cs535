import math
import numpy as np
from scipy.stats import laplace 
import matplotlib.pyplot as plt 


def gaussian_kernel(sigma, x, xi): 
    numerator = math.exp(-1.0 * (x - xi)**2 / (2 * sigma**2))
    denominator = math.sqrt(2 * math.pi * sigma ** 2) 
    return numerator / denominator


def foo(data, sigma, x):
    total = 0
    for xi in data: 
        total += gaussian_kernel(sigma, x, xi)
    return (1.0 / 1000) * total 



def get_data_y(data_x, sigma): 
    return [foo(data_x, sigma, x) for x in data_x]



sigmas = np.linspace(0.01, 1.0, 9)
data_x = sorted([np.random.laplace() for i in range(1000)])
real_laplace = np.linspace(laplace.ppf(0.01), laplace.ppf(0.99), 100)

fig, ax = plt.subplots(nrows=3, ncols=3)

i = 0
for row in ax: 
    for col in row: 
        data_y = get_data_y(data_x, sigmas[i])
        col.plot(data_x, data_y, markersize=2)
        col.plot(real_laplace, laplace.pdf(real_laplace), markersize=3)
        col.set_title('sigma = ' + str(sigmas[i]))
        i += 1


plt.show()
