import numpy as np
def z_calculation(weights, x, bias):
    z = np.dot(weights, x) + bias
    return z

def activation_function(z): # 1/(1+exp(-z))
    a = 1/(1+np.exp(-z))
    return a
