import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, input_dimension):
        self.weights = np.random.randn(input_dimension, 1)
        self.bias = np.random.randn()
