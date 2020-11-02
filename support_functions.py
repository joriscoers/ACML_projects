import numpy as np


def z_calculation(weights, x, bias):
    z = np.dot(weights, x) + bias
    return z


def activation_function(z):  # 1/(1+exp(-z))
    a = 1 / (1 + np.exp(-z))
    return a


def calculate_cost(predicted_output, training_output):
    cost = -np.sum(np.multiply(training_output, np.log(predicted_output)) + np.multiply((1 - training_output), np.log(1 - predicted_output))) / training_output.shape[1]
    return cost

