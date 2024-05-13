import numpy as np
import pandas as pd
import math


class LogisticRegression:

    def __init__(self, input):
        self.sample_size = input.shape[0]
        self.likelihood = 0
        if len(input.shape) == 2:
            self.parameters = np.ones(input.shape[1] + 1)
        else:
            self.parameters = np.ones(2)
        self.learning_rate_suppression = 'const'
        self.start_learning_rate = 1

    def correctInput(self, input):
        if len(input.shape) == 1:
            input = np.array([[value] for value in input])
        return input

    def addNeutralTermColumn(self, input):
        return np.hstack((np.ones((input.shape[0], 1)), input))

    def calculateLikelihood(self, input, output):
        if input.shape[0] != output.shape[0]:
            raise Exception(BaseException, "Input and output have to be of the same length!")
        # input has to have additional column filled with 1's for intercept
        linear_part = np.matmul(input, self.parameters)
        print(self.parameters)
        logs = np.log(1 + np.exp(linear_part))
        self.likelihood = np.exp(np.sum([output[i] * linear_part[i] for i in range(len(linear_part))] - logs))

    def calculateGradient(self, input, output, parameters):
        # input has to have additional column filled with 1's for intercept
        gradient = np.zeros((input.shape[1]))
        # for i in range(input.shape[0]):
        #     print(np.dot(input[i], parameters))
        exponents = np.array([np.exp(np.dot(input[i], parameters)) for i in range(input.shape[0])])
        exp_fractions = [1 if exponent == np.inf else exponent / (1 + exponent) for exponent in exponents]
        second_terms = [output[i] - exp_fractions[i] for i in range(len(output))]
        # for i in range(10):
        #     print(exponents[i], exp_fractions[i], second_terms[i])
        gradient = np.matmul(np.transpose(input), second_terms)
        return gradient

    def maximizeLikelihood(self, input, output, num_of_iterartions):
        likelihood_difference = np.infty
        learning_rates = [self.start_learning_rate for iteration in range(num_of_iterartions)]
        if not self.learning_rate_suppression in ['const', 'lin', 'exp']:
            raise Exception(BaseException, "Unknown learning rate suppresion method!")
        if self.start_learning_rate <= 0:
            raise Exception(BaseException, "Learning rate has to greater than 0!")
        match self.learning_rate_suppression:
            case 'lin':
                learning_rates = [self.start_learning_rate * (num_of_iterartions - iteration) / num_of_iterartions
                                  for iteration in range(num_of_iterartions)]
            case 'exp':
                learning_rates = [math.exp(-self.start_learning_rate * iteration)
                                  for iteration in range(num_of_iterartions)]
        for learning_rate in learning_rates:
            gradient = self.calculateGradient(input, output, self.parameters)
            # gradient = direction of biggest growth
            self.parameters = self.parameters + learning_rate * gradient
            previous_likelihood = self.likelihood
            self.calculateLikelihood(input, output)
            likelihood_difference = abs(previous_likelihood - self.likelihood)

    def train(self, input, output, num_of_iterartions: int = 10_000):
        self.correctInput(input)
        model_input = self.addNeutralTermColumn(input)
        self.maximizeLikelihood(model_input, output, num_of_iterartions)

    def calculateProbabilities(self, input):
        self.correctInput(input)
        proper_input = self.addNeutralTermColumn(input)
        probabilities = [1 / (1 + math.exp(-np.dot(row, self.parameters))) for row in proper_input]
        return np.array([(1 - probability,probability) for probability in probabilities])

    def predict(self, input):
        self.correctInput(input)
        probabilities = self.calculateProbabilities()
        return np.array([0 if row[0] > row[1] else 1 for row in probabilities])

df = pd.read_csv('installments.csv', sep=";")
df["installment"] = df['installment'].apply(lambda x: 1 if x=="T" else 0)
normalized_df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

X_data = normalized_df[["income", "age", "expenses"]]
y_data = normalized_df["installment"]
print(y_data)
official_reg = LogisticRegression(X_data)
official_reg.start_learning_rate = 0.01
official_reg.learning_rate_suppression = 'const'
# official_reg.train(X_data, y_data)
# print(official_reg.likelihood)
official_reg.parameters = np.array([  1.02046322, -11.44909234,  0.02744099,  9.93447708])
print(official_reg.calculateProbabilities(X_data.loc[1].to_numpy()).shape)