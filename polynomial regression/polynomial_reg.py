import numpy as np
import copy

class PolynomialRegression:

    def __init__(self, input, output):
        self.X_size = 1
        self.sample_size = input.shape[0]
        self.pol_degree = 1
        if len(input.shape) == 2:
            self.X_size = input.shape[1]
        self.coefficients = np.zeros((self.pol_degree * self.X_size +1,))

    def setPolDegree(self, pol_degree: int = 1):
        if type(pol_degree) != int:
            raise Exception(TypeError, "Degree of a polynomial has to be an integer")
        elif pol_degree < 1:
            raise Exception(ValueError, "Degree of a polynomial has to be positive")
        elif pol_degree >= self.sample_size:
            raise BaseException("Degree of polynomial can not be greater than the size of sample")
        self.pol_degree = pol_degree

    def reshapeInput(self, input):
        """reshapes input to be the correct size for further calculations"""
        return input.reshape(self.sample_size, self.X_size)

    def reshapeOutput(self, output):
        return output.reshape(self.sample_size, 1)

    def expandInput(self, input):
        """function takes as an input correctly shaped array"""
        if self.pol_degree > 1:
            for degree in range(1, self.pol_degree):
                additional_input = copy.copy(input)
                input = np.hstack((input, np.power(additional_input, degree + 1)))
        return input

    def addNeutralTermColumn(self, input):
        return np.hstack((np.ones((self.sample_size, 1)), input))

    def calculateX_squared(self, input, transposed_input):
        return np.matmul(transposed_input, input)

    def train(self, input, output):
        reg_input = self.reshapeInput(input)
        reg_input = self.expandInput(reg_input)
        reg_input = self.addNeutralTermColumn(reg_input)
        transposed_X = np.transpose(reg_input)
        inverse_X_squared = np.linalg.inv(self.calculateX_squared(reg_input, transposed_X))
        self.coefficients = np.matmul(np.matmul(inverse_X_squared, transposed_X), output)\
            .reshape((self.pol_degree * self.X_size + 1,))
        return self.coefficients


input = np.array([[1,1], [3,1], [11,4], [0,0], [2,4], [20,4] ])# ])
output = np.array([3, 7, 23, 1, 5, 41 ])# ])
print(input.shape)
regr = PolynomialRegression(input, output)
print(input)
regr.train(input, output)
print(regr.coefficients)
